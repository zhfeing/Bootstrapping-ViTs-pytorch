from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class CELoss(Loss):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)

    def forward(self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pred = output["pred"]
        gt = target["label"]
        ret = OrderedDict()
        ret["cls"] = self.loss_fn(pred, gt)
        return ret


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T: float, reduction: str = "batchmean"):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction=self.reduction) * (self.T**2)
        return loss

