from typing import Callable

import torch
import torch.nn as nn

from .norm import Norm_fn


def get_dropout(dropout: float = None):
    return nn.Dropout(dropout) if dropout is not None else nn.Identity()


def get_activation_fn(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    __SUPPORTED_ACTIVATION__ = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "glu": nn.GLU
    }
    return __SUPPORTED_ACTIVATION__[activation_name]()

