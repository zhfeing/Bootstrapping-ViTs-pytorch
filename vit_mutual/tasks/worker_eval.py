import os
import logging
from logging.handlers import QueueHandler
from typing import Dict, Any
import yaml

import torch
from torch import nn
import torch.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn

import cv_lib.utils as utils
import cv_lib.distributed.utils as dist_utils

from vit_mutual.data import build_train_dataset
from vit_mutual.models import get_model, ModelWrapper
from vit_mutual.loss import get_loss_fn
import vit_mutual.utils as vit_utils
from vit_mutual.eval import Evaluation


class Evaluator:
    def __init__(
        self,
        train_cfg: Dict[str, Any],
        val_loader: DataLoader,
        model: nn.Module,
        evaluator: Evaluation,
        distributed: bool,
        device: torch.device,
        resume: str = "",
    ):
        # set up logger
        self.logger = logging.getLogger("trainer_rank_{}".format(dist_utils.get_rank()))

        self.train_cfg = train_cfg
        self.val_loader = val_loader
        self.model = model
        self.evaluator = evaluator
        self.distributed = distributed
        self.device = device

        self.resume(resume)
        self.logger.info("Start evaluating")

    def resume(self, resume_fp: str = ""):
        """
        Resume training from checkpoint
        """
        resume_fp = os.path.expandvars(os.path.expanduser(resume_fp))
        # not a valid file
        if not os.path.isfile(resume_fp):
            return
        ckpt = torch.load(resume_fp, map_location="cpu")

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            real_model = self.model.module
        else:
            real_model = self.model
        if isinstance(real_model, ModelWrapper):
            real_model = real_model.module

        ckpt_model = ckpt["model"] if "model" in ckpt else ckpt
        real_model.load_state_dict(ckpt_model)
        self.logger.info("Loaded ckpt with epoch: %d, iter: %d", ckpt.get("epoch", -1), ckpt.get("iter", -1))

    def validate(self):
        self.logger.info("Start evaluation")
        eval_dict = self.evaluator(self.model)

        if dist_utils.is_main_process():
            self.logger.info("evaluation done")
            loss = eval_dict["loss"]
            loss_dict = eval_dict["loss_dict"]
            loss_dict = utils.tensor_dict_items(loss_dict, ndigits=4)
            acc_dict: Dict[int, float] = utils.tensor_dict_items(eval_dict["acc"], ndigits=4)
            acc_top_1 = acc_dict[1]
            acc_top_5 = acc_dict[5]
            # write logger
            info = "Validation loss: {:.5f}, acc@1: {:.4f}, acc@5: {:.4f}\nloss dict: {}"
            info = info.format(
                loss,
                acc_top_1, acc_top_5,
                utils.to_json_str(loss_dict)
            )
            self.logger.info(info)

    def __call__(self):
        self.validate()


def eval_worker(
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
    model_cfg: Dict[str, Any] = global_cfg["model"]
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
    _, val_loader, n_classes = build_train_dataset(
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args,
    )
    # create model
    logger.info("Building model...")
    model = get_model(model_cfg, n_classes)
    model.to(device)
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])

    loss = get_loss_fn(loss_cfg)
    loss.to(device)

    evaluation = Evaluation(
        loss_fn=loss,
        val_loader=val_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device,
        top_k=(1, 5)
    )
    evaluator = Evaluator(
        train_cfg=train_cfg,
        val_loader=val_loader,
        model=model,
        evaluator=evaluation,
        distributed=distributed,
        device=device,
        resume=resume
    )
    evaluator()

