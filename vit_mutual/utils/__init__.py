from typing import Dict, Tuple

import torch

from .dist_utils import LogArgs, DistLaunchArgs
from .decay_strategy import DecayStrategy
from .model import load_pretrain_model


def move_data_to_device(
    x: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = x.to(device)
    for k, v in targets.items():
        if isinstance(v, torch.Tensor):
            targets[k] = v.to(device)
    return x, targets
