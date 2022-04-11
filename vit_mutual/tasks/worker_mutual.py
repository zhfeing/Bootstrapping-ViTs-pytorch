import os
import logging
import shutil
import time
from collections import defaultdict, OrderedDict
from logging.handlers import QueueHandler
from typing import Dict, Any, List, Tuple
import datetime
import yaml

import torch
from torch import nn, Tensor
import torch.cuda
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp

import cv_lib.utils as utils
from cv_lib.config_parsing import get_tb_writer, get_cfg
from cv_lib.optimizers import get_optimizer
from cv_lib.schedulers import get_scheduler
import cv_lib.distributed.utils as dist_utils

from vit_mutual.data import build_train_dataset
from vit_mutual.models.mutual import get_mutual_model, MutualModel
from vit_mutual.loss import get_loss_fn, Loss
import vit_mutual.utils as vit_utils
from vit_mutual.eval import MutualEvaluation


class MutualTrainer:
    def __init__(
        self,
        train_cfg: Dict[str, Any],
        mutual_cfg: Dict[str, Any],
        log_args: vit_utils.LogArgs,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizers: List[Optimizer],
        lr_schedulers: List[_LRScheduler],
        model: MutualModel,
        master_model_name: str,
        loss: Loss,
        loss_weights: Dict[str, float],
        evaluator: MutualEvaluation,
        distributed: bool,
        device: torch.device,
        resume: str = "",
        use_amp: bool = False
    ):
        # set up logger
        self.logger = logging.getLogger("trainer_rank_{}".format(dist_utils.get_rank()))

        # only write in master process
        self.tb_writer = None
        if dist_utils.is_main_process():
            self.tb_writer, _ = get_tb_writer(log_args.logdir, log_args.filename)
        dist_utils.barrier()

        self.train_cfg = train_cfg
        self.mutual_cfg = mutual_cfg
        self.start_epoch = 0
        self.epoch = 0
        self.total_epoch = self.train_cfg["train_epochs"]
        self.iter = 0
        self.step = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_step = len(self.train_loader)
        self.total_iter = self.total_step * self.total_epoch
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.model = model
        self.master_model_name = master_model_name
        self.loss = loss
        self.loss_weights = loss_weights
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device
        self.ckpt_path = log_args.ckpt_path
        self.amp = use_amp
        # loss weight decay
        decay_cfg: Dict[str, Any] = self.train_cfg.get("decay_strategy", dict())
        use_decay = decay_cfg.get("use_decay", False)
        decay_items = decay_cfg.get("decay_items", list())
        self.decay_strategy = vit_utils.DecayStrategy(self.total_epoch, decay_items, use_decay)
        if use_decay:
            self.logger.info("Use decay strategy")
        # best index
        self.best_acc = 0
        self.best_iter = 0

        # for pytorch amp
        self.scalers: List[amp.GradScaler] = None
        if self.amp:
            self.logger.info("Using AMP train")
            self.scalers = list(amp.GradScaler() for _ in range(len(self.optimizers)))
        self.resume(resume)

        self.parameters_by_model = OrderedDict()
        model_without_ddp = self.model.module if isinstance(self.model, nn.parallel.DistributedDataParallel) else self.model
        for k, v in model_without_ddp.models.items():
            self.parameters_by_model[k] = v.parameters()

        self.logger.info("Start training for %d epochs", self.train_cfg["train_epochs"] - self.start_epoch)

    def resume(self, resume_fp: str = ""):
        """
        Resume training from checkpoint
        """
        # not a valid file
        if not os.path.isfile(resume_fp):
            return
        ckpt = torch.load(resume_fp, map_location="cpu")

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])
        for i, (optim_state, scheduler_state) in enumerate(zip(ckpt["optimizers"], ckpt["lr_schedulers"])):
            self.optimizers[i].load_state_dict(optim_state)
            self.lr_schedulers[i].load_state_dict(scheduler_state)
        # load grad scalers
        if self.scalers is not None and "grad_scaler" in ckpt:
            for i, scaler_state in enumerate(ckpt["grad_scalers"]):
                self.scalers[i].load_state_dict(scaler_state)
        self.iter = ckpt["iter"] + 1
        self.start_epoch = ckpt["epoch"] + 1
        self.logger.info("Loaded ckpt with epoch: %d, iter: %d", ckpt["epoch"], ckpt["iter"])

    def train_iter(self, x: Tensor, targets: List[Dict[str, Any]]):
        self.model.train()
        self.loss.train()
        # move to device
        x, targets = vit_utils.move_data_to_device(x, targets, self.device)

        [optimizer.zero_grad() for optimizer in self.optimizers]
        with amp.autocast(enabled=self.amp):
            output = self.model(x)
            loss_dict: Dict[str, torch.Tensor] = self.loss(output, targets)
            # split prefix to different model losses
            _weighted_losses: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
            decay_dict: Dict[str, float] = dict()
            for k, loss in loss_dict.items():
                model_name, k_prefix = k.split(".")[0:2]
                new_k = k[len(model_name) + 1:]
                if k_prefix in self.loss_weights:
                    weight = self.loss_weights[k_prefix]
                    decay = self.decay_strategy(self.epoch, k_prefix)
                    decay_dict[k_prefix] = torch.tensor(decay)
                    # decay weight
                    _weighted_losses[model_name][new_k] = loss * weight * decay
            weighted_losses: Dict[str, Dict[str, torch.Tensor]] = OrderedDict()
            losses: Dict[str, torch.Tensor] = OrderedDict()
            for k in sorted(_weighted_losses):
                weighted_losses[k] = _weighted_losses[k]
                losses[k] = sum(weighted_losses[k].values())

        if self.amp:
            for i, (model_name, loss) in enumerate(losses.items()):
                self.scalers[i].scale(loss).backward()
                # grad clip
                clip_grad = self.mutual_cfg["cfg"][model_name].get("clip_max_norm", None)
                if clip_grad is not None:
                    if model_name == "joint":
                        # Un-scales the gradients of optimizer's assigned params in-place
                        self.scalers[i].unscale_(self.optimizers[i])
                        self.optimizers[i].param_groups
                        nn.utils.clip_grad.clip_grad_norm_(
                            self.parameters_by_model[self.mutual_cfg["master"]],
                            clip_grad
                        )
                    else:
                        # Un-scales the gradients of optimizer's assigned params in-place
                        self.scalers[i].unscale_(self.optimizers[i])
                        self.optimizers[i].param_groups
                        nn.utils.clip_grad.clip_grad_norm_(
                            self.parameters_by_model[model_name],
                            clip_grad
                        )
                self.scalers[i].step(self.optimizers[i])
                self.scalers[i].update()
        else:
            for i, (model_name, loss) in enumerate(losses.items()):
                loss.backward()
                # grad clip
                clip_grad = self.mutual_cfg["cfg"][model_name].get("clip_max_norm", None)
                if clip_grad is not None:
                    if model_name == "joint":
                        nn.utils.clip_grad.clip_grad_norm_(
                            self.parameters_by_model[self.mutual_cfg["master"]],
                            clip_grad
                        )
                    else:
                        nn.utils.clip_grad.clip_grad_norm_(
                            self.parameters_by_model[model_name],
                            clip_grad
                        )
                self.optimizers[i].step()

        # save memory
        [optimizer.zero_grad(set_to_none=True) for optimizer in self.optimizers]
        for k, v in losses.items():
            losses[k] = dist_utils.reduce_tensor(v.detach())
        loss_dict = dist_utils.reduce_dict(loss_dict)
        # print
        if self.iter % self.train_cfg["print_interval"] == 0 and dist_utils.is_main_process():
            losses = utils.tensor_dict_items(losses, ndigits=4)
            loss_dict = utils.tensor_dict_items(loss_dict, ndigits=4)
            lr_dict = {f"{idx}": scheduler.get_last_lr()[0] for idx, scheduler in enumerate(self.lr_schedulers)}
            decay_dict = utils.tensor_dict_items(decay_dict, ndigits=4)
            # reduce loss
            self.logger.info(
                "Epoch %3d|%3d, step %4d|%4d, iter %5d|%5d, lr: %s, loss: %s,\nloss dict: %s\ndecay dict: %s",
                self.epoch, self.total_epoch,
                self.step, self.total_step,
                self.iter, self.total_iter,
                utils.to_json_str(list(lr_dict.values()), indent=None),
                utils.to_json_str(losses, indent=None),
                utils.to_json_str(loss_dict),
                utils.to_json_str(decay_dict)
            )
            self.tb_writer.add_scalars("Train/Loss", losses, self.iter)
            self.tb_writer.add_scalars("Train/Loss_dict", loss_dict, self.iter)
            self.tb_writer.add_scalars("Train/Lr", lr_dict, self.iter)
            if len(decay_dict) > 0:
                self.tb_writer.add_scalars("Train/Decay", decay_dict, self.iter)
        dist_utils.barrier()
        self.iter += 1

    def validate_and_save(self, show_tb: bool = True):
        self.logger.info("Start evaluation")
        eval_dict = self.evaluator(self.model)
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        if dist_utils.is_main_process():
            self.logger.info("evaluation done")
            losses = eval_dict["loss"]
            loss_dict = eval_dict["loss_dict"]
            losses = utils.tensor_dict_items(losses, ndigits=5)
            loss_dict = utils.tensor_dict_items(loss_dict, ndigits=4)
            acc1_dict: Dict[str, float] = utils.tensor_dict_items(eval_dict["acc1"], ndigits=4)
            acc5_dict: Dict[str, float] = utils.tensor_dict_items(eval_dict["acc5"], ndigits=4)
            # write logger
            info = "Validation loss: {}, acc@1: {}, acc@5: {}\nloss dict: {}"
            info = info.format(
                utils.to_json_str(losses, indent=None),
                utils.to_json_str(acc1_dict, indent=None),
                utils.to_json_str(acc5_dict, indent=None),
                utils.to_json_str(loss_dict),
            )
            self.logger.info(info)
            acc_top_1 = acc1_dict[self.master_model_name]
            if show_tb:
                # write tb logger
                self.tb_writer.add_scalars("Val/Loss", losses, self.iter)
                self.tb_writer.add_scalars("Val/Acc@1", acc1_dict, self.iter)
                self.tb_writer.add_scalars("Val/Acc@5", acc5_dict, self.iter)
                self.tb_writer.add_scalars("Val/Loss_dict", loss_dict, self.iter)
                self.tb_writer.add_scalar("Val/MasterAcc@1", acc_top_1, self.iter)

            # save ckpt
            state_dict = {
                "model": model_state_dict,
                "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
                "lr_schedulers": [scheduler.state_dict() for scheduler in self.lr_schedulers],
                "epoch": self.epoch,
                "iter": self.iter,
                "eval_dict": eval_dict,
                "loss_dict": loss_dict
            }
            if self.scalers is not None:
                state_dict["grad_scalers"] = [scaler.state_dict() for scaler in self.scalers]
            save_fp = os.path.join(self.ckpt_path, f"iter-{self.iter}.pth")
            self.logger.info("Saving state dict to %s...", save_fp)
            torch.save(state_dict, save_fp)
            if acc_top_1 > self.best_acc:
                # best index
                self.best_acc = acc_top_1
                self.best_iter = self.iter
                shutil.copy(save_fp, os.path.join(self.ckpt_path, "best.pth"))
        dist_utils.barrier()

    def __call__(self, debug: bool = False):
        start_time = time.time()
        if not debug:
            self.validate_and_save(show_tb=False)
        # start one epoch
        for self.epoch in range(self.start_epoch, self.train_cfg["train_epochs"]):
            if self.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            for self.step, (x, target) in enumerate(self.train_loader):
                self.train_iter(x, target)
                # validation
                if self.iter % self.train_cfg["val_interval"] == 0:
                    self.validate_and_save()
            [scheduler.step() for scheduler in self.lr_schedulers]
        self.logger.info("Final validation")
        self.validate_and_save()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if dist_utils.is_main_process():
            self.logger.info("Training time %s", total_time_str)
            self.logger.info("Best acc: %f, iter: %d", self.best_acc, self.best_iter)


def get_mutual_learners(
    train_cfg: Dict[str, Any],
    mutual_cfg: Dict[str, Any],
    mutual_models: MutualModel
) -> Tuple[List[Optimizer], List[_LRScheduler]]:
    logger = logging.getLogger("get_mutual_learners")
    if mutual_cfg["joint_training"]:
        # the sharing weights will not be handled properly
        optimizer = get_optimizer(mutual_models.parameters(), train_cfg["optimizer"])
        logger.info("Loaded optimizer:\n%s", optimizer)
        lr_scheduler = get_scheduler(optimizer, train_cfg["lr_schedule"])
        return [optimizer], [lr_scheduler]
    # one optimizerand lr scheduler for each model
    optimizers = list()
    lr_schedulers = list()
    for name, model in mutual_models.models.items():
        cfg = get_cfg(mutual_cfg["cfg"][name]["cfg_path"])
        optimizer = get_optimizer(model.parameters(), cfg["training"]["optimizer"])
        optimizers.append(optimizer)
        lr_schedulers.append(get_scheduler(optimizer, cfg["training"]["lr_schedule"]))
    return optimizers, lr_schedulers


def mutual_worker(
    gpu_id: int,
    launch_args: vit_utils.DistLaunchArgs,
    log_args: vit_utils.LogArgs,
    global_cfg: Dict[str, Any],
    resume: str = ""
):
    """
    What created in this function is only used in this process and not shareable
    """
    # setup process root logger
    if launch_args.distributed:
        root_logger = logging.getLogger()
        handler = QueueHandler(log_args.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    # split configs
    data_cfg: Dict[str, Any] = global_cfg["dataset"]
    train_cfg: Dict[str, Any] = global_cfg["training"]
    val_cfg: Dict[str, Any] = global_cfg["validation"]
    mutual_cfg: Dict[str, Any] = global_cfg["mutual"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
    # set debug number of workers
    if launch_args.debug:
        train_cfg["num_workers"] = 0
        val_cfg["num_workers"] = 0
        train_cfg["print_interval"] = 1
        train_cfg["val_interval"] = 10
    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            current_rank = launch_args.rank * launch_args.ngpus_per_node + gpu_id
        dist.init_process_group(
            backend=launch_args.backend,
            init_method=launch_args.master_url,
            world_size=launch_args.world_size,
            rank=current_rank
        )

    assert dist_utils.get_rank() == current_rank, "code bug"
    # set up process logger
    logger = logging.getLogger("worker_rank_{}".format(current_rank))

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        utils.make_deterministic(seed)
    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    # get dataloader
    logger.info("Building dataset...")
    train_loader, val_loader, n_classes = build_train_dataset(
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args,
    )
    # create model
    logger.info("Building model...")
    mutual_model = get_mutual_model(mutual_cfg["cfg"], n_classes)
    mutual_model.to(device)
    model_without_ddp = mutual_model
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            mutual_model = nn.SyncBatchNorm.convert_sync_batchnorm(mutual_model)
        mutual_model = nn.parallel.DistributedDataParallel(mutual_model, device_ids=[gpu_id])
        model_without_ddp = mutual_model.module

    optimizers, lr_schedulers = get_mutual_learners(train_cfg, mutual_cfg, model_without_ddp)

    loss = get_loss_fn(loss_cfg)
    loss.to(device)

    evaluator = MutualEvaluation(
        loss_fn=loss,
        val_loader=val_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device
    )

    trainer = MutualTrainer(
        train_cfg=train_cfg,
        mutual_cfg=mutual_cfg,
        log_args=log_args,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        model=mutual_model,
        master_model_name=mutual_cfg["master"],
        loss=loss,
        loss_weights=loss_cfg["weight_dict"],
        evaluator=evaluator,
        distributed=distributed,
        device=device,
        resume=resume,
        use_amp=launch_args.use_amp
    )
    # start training
    trainer(launch_args.debug)

