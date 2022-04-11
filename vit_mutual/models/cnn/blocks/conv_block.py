import math
from copy import deepcopy
from typing import Callable, Any, Dict, Tuple

import torch
import torch.nn as nn
from vit_mutual.models.cnn.blocks.conv import Conv_2d, conv_2d


############################################################################################
# Conv Block for replacing MHSA

class Conv_SL(nn.Module):
    def __init__(self, embed_dim: int, bias: bool = True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv_ManyHead(nn.Module):
    """
    General convolution layer with customized reception field
    """
    def __init__(self, embed_dim: int, bias: bool = True, num_heads: int = 6, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = math.ceil(math.sqrt(num_heads))

        self.weight = nn.Parameter(torch.empty(embed_dim, embed_dim, num_heads))
        self.bias = nn.Parameter(torch.empty(embed_dim)) if bias else None

        self.phi: torch.Tensor = None
        self.last_shape: Tuple[int, int] = None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_phi(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        phi = Conv_2d.get_phi(
            shape=shape,
            device=device,
            kernel_size=(self.kernel_size, self.kernel_size),
            flatten=False
        )
        phi = phi[:self.num_heads]
        phi = phi.permute(1, 0, 2).flatten(1)
        return phi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [bs, c_in, H, W]
        """
        shape = x.shape[2:]
        if self.last_shape == shape and self.phi is not None:
            phi = self.phi
        else:
            phi = self.get_phi(shape, x.device)
            self.last_shape = shape
            self.phi = phi
        return conv_2d(x, phi=phi, weights=self.weight.permute(2, 0, 1), bias=self.bias)


__REGISTERED_CONV__ = {
    "conv_sl": Conv_SL,
    "conv_manyhead": Conv_ManyHead,
}


def get_conv_block(
    embed_dim: int,
    activation: str,
    norm: Callable[[Dict[str, Any]], nn.Module],
    conv_cfg: Dict[str, Any]
):
    conv_cfg = deepcopy(conv_cfg)
    name = conv_cfg.pop("name")
    conv_block = __REGISTERED_CONV__[name](
        embed_dim=embed_dim,
        activation=activation,
        norm=norm,
        **conv_cfg
    )
    return conv_block
