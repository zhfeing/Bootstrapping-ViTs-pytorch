from typing import Optional, Tuple
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True, kernel_size: int = 3):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=bias)

    def get_weight(self, kernel_id: int = None) -> torch.Tensor:
        """
        Get W_i w.r.t. the formula in the
        Return:
            linear-like weight with shape [c_out, c_in, N] or kernel weight [c_out, c_in]
        """
        weight = self.weight.flatten(2)
        if kernel_id is not None:
            weight = weight[..., kernel_id]
        return weight

    def get_bias(self) -> Optional[torch.Tensor]:
        return self.bias

    @classmethod
    def get_phi(
        cls,
        shape: Tuple[int, int],
        device: torch.device,
        kernel_size: Tuple[int, int] = (3, 3),
        flatten: bool = True
    ) -> torch.Tensor:
        """
        Args:
            shape: [H, W]
        Return: Tensor with shape [N, n, n] if not flatten, [n, Nxn], N = k^2, n = seq_len if flatten
        """
        phi = torch.zeros(
            kernel_size[0] * kernel_size[1],
            shape[0] * shape[1],
            shape[0] * shape[1],
            device=device
        )
        for point_idx, point in enumerate(product(range(shape[0]), range(shape[1]))):
            neighbor = _get_neighborhood(point, shape[1], size=kernel_size)
            points = neighbor[:, 0:2]
            mask_dim0 = torch.logical_and(0 <= points[:, 0], points[:, 0] < shape[0])
            mask_dim1 = torch.logical_and(0 <= points[:, 1], points[:, 1] < shape[1])
            mask = torch.logical_and(mask_dim0, mask_dim1)
            pos = neighbor[:, 2]
            leagel_pos = pos[mask]
            phi[mask, point_idx, leagel_pos] = 1

        if flatten:
            # permute phi [N, n, n] -> [n, N, n] -> [n, Nxn]
            phi = phi.permute(1, 0, 2).flatten(1)
        return phi


def _get_neighborhood(point: Tuple[int, int], width: int, size: Tuple[int, int] = (3, 3)) -> torch.Tensor:
    """
    Args:
        point: (dim_0, dim_1)
        size: must be odd number
    """
    assert size[0] % 2 == 1 and size[1] % 2 == 1, "size must be odd number"
    dim_0, dim_1 = point
    delta_0 = (size[0] - 1) // 2
    delta_1 = (size[1] - 1) // 2
    neighbor_0 = torch.arange(dim_0 - delta_0, dim_0 + delta_0 + 1, dtype=torch.long)
    neighbor_1 = torch.arange(dim_1 - delta_1, dim_1 + delta_1 + 1, dtype=torch.long)
    points: torch.Tensor = torch.cartesian_prod(neighbor_0, neighbor_1)
    pos = points[:, 0] * width + points[:, 1]
    return torch.cat([points, pos[:, None]], dim=1)


def conv_2d(
    x: torch.Tensor,
    phi: torch.Tensor,
    weights: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
):
    """
    Args:
        x: Tensor with shape [bs, c_in, H, W]
        weight: Tensor with shape [N, c_out, c_in]
        bias: Tensor with shape [c_out]
        phi: Tensor with shape [HW, NxHW]
    """
    shape = x.shape[2:]
    N, c_out = weights.shape[:2]
    # permute x [bs, c_in, H, W] -> [bs, HW, c_in]
    x = x.flatten(start_dim=2).transpose(1, 2)
    bs, n = x.shape[:2]
    # permute weight [N, c_out, c_in] -> [Nxc_out, c_in]
    weights = weights.flatten(0, 1)
    # x -> y with [bs, HW, c_in] -> [bs, HW, Nxc_out] -> [bs, HW, N, c_out]
    # y = F.linear(x, weights, bias=None).unflatten(dim=-1, sizes=(N, c_out))
    y = F.linear(x, weights, bias=None).view(bs, n, N, c_out)
    # permute y [bs, HW, N, c_out] -> [bs, N, HW, c_out] -> [bs, NxHW, c_out]
    y = y.permute(0, 2, 1, 3).flatten(1, 2)
    # sum [bs, c_out, HW]
    y: torch.Tensor = torch.einsum("nm, bmc -> bcn", phi, y)
    if bias is not None:
        y += bias[None, :, None]
    # [bs, C_out, HW] -> [bs, C_out, H, W]
    # y = y.unflatten(dim=-1, sizes=shape)
    y = y.view(bs, c_out, shape[0], shape[1])
    return y

