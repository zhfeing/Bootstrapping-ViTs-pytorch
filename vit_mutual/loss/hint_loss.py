from collections import OrderedDict
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoResample(nn.Module):
    def __init__(self, enable: bool = True):
        super().__init__()
        self.enable = enable

    def forward(self, feat_1: torch.Tensor, feat_2: torch.Tensor):
        """
        Args:
            feat_1: [n1, bs, emb_dim]
            feat_2: [n2, bs, emb_dim]
        """
        if not self.enable:
            return feat_1, feat_2

        shape_1 = feat_1.shape
        shape_2 = feat_2.shape
        if shape_1 == shape_2:
            return feat_1, feat_2

        assert shape_1[1:] == shape_2[1:], "bs and emb_dim must be equal"
        exchange = False
        if shape_1[0] > shape_2[0]:
            feat_long = feat_1
            feat_short = feat_2
            assert shape_1[0] % shape_2[0] == 0
            step = shape_1[0] // shape_2[0]
        else:
            exchange = True
            feat_long = feat_2
            feat_short = feat_1
            assert shape_2[0] % shape_1[0] == 0
            step = shape_2[0] // shape_1[0]

        # [n, bs, emb_dim] -> [bs, emb_dim, n]
        feat_long = feat_long.permute(1, 2, 0)
        feat_long = F.avg_pool1d(feat_long, kernel_size=step, stride=step)
        # [bs, emb_dim, n] -> [n, bs, emb_dim]
        feat_long = feat_long.permute(2, 0, 1)

        if exchange:
            feat_long, feat_short = feat_short, feat_long
        return feat_long, feat_short


class DistillHint(nn.Module):
    def __init__(
        self,
        start_id: int = 1,
        proj_student: bool = False,
        embed_dim: int = None,
        norm: bool = False,
        auto_resample: bool = False
    ):
        """
        Args:
            start_id: 1 for one [cls] token, 2 for one [cls] and one [dist] token
        """
        super().__init__()
        self.start_id = start_id
        self.proj = nn.Identity()
        self.resample = AutoResample(auto_resample)
        self.norm_s = nn.Identity()
        self.norm_t = nn.Identity()
        self.mse_loss = nn.MSELoss()

        if proj_student:
            self.proj = nn.LazyLinear(embed_dim)
        if norm:
            self.norm_s = nn.LayerNorm(embed_dim)
            self.norm_t = nn.LayerNorm(embed_dim)

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        if feat_s.dim() == 3:
            assert feat_t.dim() == 4
            feat_cnn = feat_t
            feat_vit = feat_s
        elif feat_s.dim() == 4:
            assert feat_t.dim() == 3
            feat_cnn = feat_s
            feat_vit = feat_t
        else:
            raise Exception(f"Student feature shape {feat_s.shape} is not supported")

        # [n + 1(2), bs, emb_dim] -> [n, bs, emb_dim]
        feat_vit = feat_vit[self.start_id:]
        # [bs, emb_dim, w, h] -> [wh, bs, emb_dim]
        feat_cnn = feat_cnn.flatten(2).permute(2, 0, 1)

        # proj student
        feat_vit = self.proj(feat_vit)
        feat_cnn, feat_vit = self.resample(feat_cnn, feat_vit)
        assert feat_vit.shape == feat_cnn.shape

        feat_vit = self.norm_s(feat_vit)
        feat_cnn = self.norm_t(feat_cnn)

        loss = self.mse_loss(feat_vit, feat_cnn)
        return loss
