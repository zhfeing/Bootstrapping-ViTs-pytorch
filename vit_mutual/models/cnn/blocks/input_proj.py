from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn

from vit_mutual.models import layers


############################################################################################
# Input Proj

class VitInputProj(nn.Conv2d):
    def __init__(
        self,
        image_channels: int,
        embed_dim: int,
        patch_size: int = 16
    ):
        super().__init__(
            in_channels=image_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.weight)
        nn.init.zeros_(self.bias)


class ResnetInputProj(nn.Module):
    def __init__(
        self,
        image_channels: int,
        embed_dim: int,
        activation: str,
        mid_channel: int = 64,
        **kwargs
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, mid_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(num_features=mid_channel)
        self.activation = layers.get_activation_fn(activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_channel, embed_dim, kernel_size=3, stride=2, padding=1, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.conv1.bias is not None:
            nn.init.normal_(self.conv1.bias, 1e-6)
        if self.conv2.bias is not None:
            nn.init.normal_(self.conv2.bias, 1e-6)

    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2(x)
        return x


__REGISTERED_INPUT_PROJ__ = {
    "vit_like": VitInputProj,
    "resnet_like": ResnetInputProj,
}


def get_input_proj(
    embed_dim: int,
    proj_cfg: Dict[str, Any]
):
    proj_cfg = deepcopy(proj_cfg)
    name = proj_cfg.pop("name")
    proj = __REGISTERED_INPUT_PROJ__[name](
        embed_dim=embed_dim,
        **proj_cfg
    )
    return proj

