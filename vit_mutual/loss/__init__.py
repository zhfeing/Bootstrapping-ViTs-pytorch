from copy import deepcopy
from typing import Any, Dict

from .base_loss import CELoss, Loss, DistillKL
from .hint_loss import DistillHint
from .mutual_loss import BaseMutualLoss, JointMutualLoss


__REGISTERED_LOSS__ = {
    "ce_loss": CELoss,
    "base_mutual_loss": BaseMutualLoss,
    "joint_mutual_loss": JointMutualLoss,
}


def get_loss_fn(loss_cfg: Dict[str, Any]) -> Loss:
    loss_cfg = deepcopy(loss_cfg)
    name = loss_cfg.pop("name")
    loss_cfg.pop("weight_dict")
    return __REGISTERED_LOSS__[name](**loss_cfg)

