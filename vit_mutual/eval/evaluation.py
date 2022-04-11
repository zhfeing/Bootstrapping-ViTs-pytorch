from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Tuple
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv_lib.distributed.utils as dist_utils
import cv_lib.metrics as metrics

from vit_mutual.loss import Loss
from vit_mutual.utils import move_data_to_device


class Evaluation:
    """
    Distributed classification evaluator
    """
    def __init__(
        self,
        loss_fn: Loss,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device,
        top_k: Tuple[int] = (1,)
    ):
        self.main_process = dist_utils.is_main_process()
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self.val_loader = val_loader
        self.device = device
        self.top_k = top_k

    def get_loss(self, output: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        loss_dict: Dict[str, torch.Tensor] = self.loss_fn(output, targets)
        weighted_losses: Dict[str, torch.Tensor] = dict()
        for k, loss in loss_dict.items():
            k_prefix = k.split(".")[0]
            if k_prefix in self.loss_weights:
                weighted_losses[k] = loss * self.loss_weights[k_prefix]
        loss = sum(weighted_losses.values())
        loss = loss.detach()
        return loss, loss_dict

    def __call__(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Return:
            dictionary:
            {
                loss:
                loss_dict:
                performance:
            }
        """
        model.eval()
        self.loss_fn.eval()

        loss_meter = metrics.AverageMeter()
        loss_dict_meter = metrics.DictAverageMeter()
        acc_meter = metrics.DictAverageMeter()
        # only show in main process
        tqdm_shower = None
        if self.main_process:
            tqdm_shower = tqdm.tqdm(total=len(self.val_loader), desc="Val Batch")

        with torch.no_grad():
            for samples, targets in self.val_loader:
                samples, targets = move_data_to_device(samples, targets, self.device)
                output = model(samples)
                # calculate loss
                loss, loss_dict = self.get_loss(output, targets)
                loss_meter.update(loss)
                loss_dict_meter.update(loss_dict)
                # calculate acc
                acc_top_k = metrics.accuracy(output["pred"], targets["label"], self.top_k)
                acc_top_k = {k: acc for k, acc in zip(self.top_k, acc_top_k)}
                acc_meter.update(acc_top_k)
                # update tqdm
                if self.main_process:
                    tqdm_shower.update()
        if self.main_process:
            tqdm_shower.close()
        dist_utils.barrier()

        # accumulate
        loss_meter.accumulate()
        loss_dict_meter.accumulate()
        acc_meter.accumulate()
        loss_meter.sync()
        loss_dict_meter.sync()
        acc_meter.sync()

        ret = dict(
            loss=loss_meter.value(),
            loss_dict=loss_dict_meter.value(),
            acc=acc_meter.value()
        )
        return ret


class MutualEvaluation:
    """
    Distributed classification evaluator
    """
    def __init__(
        self,
        loss_fn: Loss,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        device: torch.device,
    ):
        self.main_process = dist_utils.is_main_process()
        self.loss_fn = loss_fn
        self.loss_weights = loss_weights
        self.val_loader = val_loader
        self.device = device
        self.top_k = (1, 5)

    def get_loss(self, output: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        loss_dict: Dict[str, torch.Tensor] = self.loss_fn(output, targets)
        weighted_losses: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        for k, loss in loss_dict.items():
            model_name, k_prefix = k.split(".")[0:2]
            new_k = k[len(model_name) + 1:]
            if k_prefix in self.loss_weights:
                weighted_losses[model_name][new_k] = loss * self.loss_weights[k_prefix]
        losses: Dict[str, torch.Tensor] = OrderedDict()
        for k, v in weighted_losses.items():
            losses[k] = sum(v.values()).detach()
        return losses, loss_dict

    def __call__(
        self,
        model: nn.Module
    ) -> Dict[str, Any]:
        """
        Return:
            dictionary:
            {
                loss:
                loss_dict:
                performance:
            }
        """
        model.eval()
        self.loss_fn.eval()

        loss_meter = metrics.DictAverageMeter()
        loss_dict_meter = metrics.DictAverageMeter()
        acc_1_meter = metrics.DictAverageMeter()
        acc_5_meter = metrics.DictAverageMeter()
        # only show in main process
        tqdm_shower = None
        if self.main_process:
            tqdm_shower = tqdm.tqdm(total=len(self.val_loader), desc="Val Batch")

        with torch.no_grad():
            for samples, targets in self.val_loader:
                samples, targets = move_data_to_device(samples, targets, self.device)
                output = model(samples)
                # calculate loss
                loss, loss_dict = self.get_loss(output, targets)
                loss_meter.update(loss)
                loss_dict_meter.update(loss_dict)
                # calculate acc
                acc_top_1 = dict()
                acc_top_5 = dict()
                for model_name, pred in output["preds"].items():
                    acc1, acc5 = metrics.accuracy(pred, targets["label"], self.top_k)
                    acc_top_1[model_name] = acc1
                    acc_top_5[model_name] = acc5
                acc_1_meter.update(acc_top_1)
                acc_5_meter.update(acc_top_5)
                # update tqdm
                if self.main_process:
                    tqdm_shower.update()
        if self.main_process:
            tqdm_shower.close()
        dist_utils.barrier()

        # accumulate
        loss_meter.accumulate()
        loss_dict_meter.accumulate()
        acc_1_meter.accumulate()
        acc_5_meter.accumulate()
        loss_meter.sync()
        loss_dict_meter.sync()
        acc_1_meter.sync()
        acc_5_meter.sync()

        ret = dict(
            loss=loss_meter.value(),
            loss_dict=loss_dict_meter.value(),
            acc1=acc_1_meter.value(),
            acc5=acc_5_meter.value()
        )
        return ret
