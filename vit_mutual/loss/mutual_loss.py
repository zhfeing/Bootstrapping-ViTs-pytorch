from typing import Dict, List, Any
from collections import OrderedDict
from itertools import product

import torch
import torch.nn.functional as F

from vit_mutual.loss import Loss, DistillKL, DistillHint


class BaseMutualLoss(Loss):
    """
    Mutual learning loss for ViT and CNNs
    """
    def __init__(
        self,
        loss_items: List[str],
        kd_temp: float = 4,
        hint_cfg: Dict[str, Any] = None
    ):
        super().__init__()
        self.loss_items = loss_items
        self.kl_div_fn = DistillKL(kd_temp)
        if hint_cfg is not None:
            self.hint_fn = DistillHint(**hint_cfg)

    def mutual_ce_loss(self, output: Dict[str, Any], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds: Dict[str, torch.Tensor] = output["preds"]
        gt = target["label"]
        ret: Dict[str, torch.Tensor] = OrderedDict()
        for model_name, pred in preds.items():
            if isinstance(pred, dict):
                pred = pred["pred"]
            ce_loss = F.cross_entropy(pred, gt)
            ret[f"{model_name}.cls"] = ce_loss
        return ret

    def mutual_kd_loss(self, output: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = OrderedDict()
        model_names = list(output["preds"])
        for m1, m2 in product(model_names, repeat=2):
            if m1 != m2:
                pred_1 = output["preds"][m1]
                pred_2 = output["preds"][m2]
                if isinstance(pred_1, dict):
                    pred_1 = pred_1["dist"]
                if isinstance(pred_2, dict):
                    pred_2 = pred_2["dist"]
                kl_div = self.kl_div_fn(pred_1, pred_2.detach())
                ret[f"{m1}.kd.{m2}"] = kl_div
        return ret

    def mutual_hint_loss(self, output: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = OrderedDict()
        features: Dict[str, torch.Tensor] = output["mid_features"]
        model_names = list(features)
        for m1, m2 in product(model_names, repeat=2):
            if m1 != m2:
                feats_1 = output["mid_features"][m1]
                feats_2 = output["mid_features"][m2]
                assert len(feats_1) == len(feats_2)
                for i, (feat_1, feat_2) in enumerate(zip(feats_1.values(), feats_2.values())):
                    loss = self.hint_fn(feat_1, feat_2.detach())
                    if feat_1.dim() == 3:
                        ret[f"{m1}.hint_vit.{m2}_layer_{i}"] = loss
                    elif feat_1.dim() == 4:
                        ret[f"{m1}.hint_cnn.{m2}_layer_{i}"] = loss
                    else:
                        raise Exception("Invalid feat dimension")
        return ret

    def get_loss(self, name: str, output: Dict[str, Any], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_map = {
            "cross_entropy": self.mutual_ce_loss,
            "mutual_kd": self.mutual_kd_loss,
            "mutual_hint": self.mutual_hint_loss,
        }
        return loss_map[name](output=output, target=target)

    def forward(self, output: Dict[str, Any], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: dict with keys
            {
                vit_pred: Tensor
                cnn_pred: Tensor
                vit_mid_features: OrderDict[str, Tensor]
                cnn_mid_features: OrderDict[str, Tensor]
            }
        """
        losses: Dict[str, torch.Tensor] = OrderedDict()
        for loss_name in self.loss_items:
            loss_item = self.get_loss(
                loss_name,
                output,
                target
            )
            losses.update(loss_item)
        return losses


class JointMutualLoss(BaseMutualLoss):
    def __init__(
        self,
        loss_items: List[str],
        kd_temp: float = 4,
        hint_cfg: Dict[str, Any] = None,
    ):
        super().__init__(loss_items, kd_temp=kd_temp, hint_cfg=hint_cfg)

    def joint_ce_loss(self, output: Dict[str, Any], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        preds: Dict[str, torch.Tensor] = output["preds"]
        gt = target["label"]
        ret: Dict[str, torch.Tensor] = OrderedDict()
        for model_name, pred in preds.items():
            if isinstance(pred, dict):
                pred = pred["pred"]
            ce_loss = F.cross_entropy(pred, gt)
            ret[f"joint.cls_{model_name}"] = ce_loss
        return ret

    def joint_kd_loss(self, output: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = OrderedDict()
        model_names = list(output["preds"])
        for m1, m2 in product(model_names, repeat=2):
            if m1 != m2:
                pred_1 = output["preds"][m1]
                pred_2 = output["preds"][m2]
                if isinstance(pred_1, dict):
                    pred_1 = pred_1["dist"]
                if isinstance(pred_2, dict):
                    pred_2 = pred_2["dist"]
                kl_div = self.kl_div_fn(pred_1, pred_2.detach())
                ret[f"joint.kd_{m1}.{m2}"] = kl_div
        return ret

    def joint_hint_loss(self, output: Dict[str, Any], **kwargs) -> Dict[str, torch.Tensor]:
        ret: Dict[str, torch.Tensor] = OrderedDict()
        features: Dict[str, torch.Tensor] = output["mid_features"]
        model_names = list(features)
        for m1, m2 in product(model_names, repeat=2):
            if m1 != m2:
                feats_1 = output["mid_features"][m1]
                feats_2 = output["mid_features"][m2]
                assert len(feats_1) == len(feats_2)
                for i, (feat_1, feat_2) in enumerate(zip(feats_1.values(), feats_2.values())):
                    loss = self.hint_fn(feat_1, feat_2.detach())
                    if feat_1.dim() == 3:
                        ret[f"joint.hint_{m1}.{m2}_layer_{i}"] = loss
                    elif feat_1.dim() == 4:
                        ret[f"joint.hint_{m1}.{m2}_layer_{i}"] = loss
                    else:
                        raise Exception("Invalid feat dimension")
        return ret

    def get_loss(self, name: str, output: Dict[str, Any], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss_map = {
            "joint_ce": self.joint_ce_loss,
            "joint_kd": self.joint_kd_loss,
            "joint_hint": self.joint_hint_loss,
        }
        return loss_map[name](output=output, target=target)

