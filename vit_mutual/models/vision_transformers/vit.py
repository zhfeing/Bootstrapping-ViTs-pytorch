from typing import List

import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .pos_encoding import PosEncoding
from vit_mutual.models.transformer import Transformer
from vit_mutual.models.transformer.transformer import MLP, MultiHeadSelfAttention


class ViT(nn.Module):
    def __init__(
        self,
        patch_embed: PatchEmbed,
        pos_embed: PosEncoding,
        transformer: Transformer,
        num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.transformer = transformer

        embed_dim = self.transformer.embed_dim
        self.cls_head = nn.Linear(self.transformer.embed_dim, num_classes)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def get_mhsa(self) -> List[MultiHeadSelfAttention]:
        mhsa = [layer.attention for layer in self.transformer.layers]
        return mhsa

    def get_mlp(self) -> List[MLP]:
        mlp = [layer.mlp for layer in self.transformer.layers]
        return mlp

    def forward(self, img: torch.Tensor):
        # seq has shape [n, bs, dim]
        seq: torch.Tensor = self.patch_embed(img)
        bs = seq.shape[1]
        cls_token = self.cls_token.expand(-1, bs, -1)
        # add cls token
        seq = torch.cat((cls_token, seq), dim=0)
        # pos embedding
        seq = self.pos_embed(seq)
        seq = self.transformer(seq)
        cls_token = seq[0]
        prob = self.cls_head(cls_token)

        return prob

