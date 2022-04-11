import torch
import torch.nn as nn

from .vit import ViT


class DeiT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embed_dim = self.transformer.embed_dim
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.dist_token, std=0.02)
        self.dist_head = nn.Linear(embed_dim, self.num_classes)

    def forward(self, img: torch.Tensor):
        # seq has shape [n, bs, dim]
        seq: torch.Tensor = self.patch_embed(img)
        bs = seq.shape[1]
        cls_token = self.cls_token.expand(-1, bs, -1)
        dist_token = self.dist_token.expand(-1, bs, -1)
        # add cls and dist token
        seq = torch.cat((cls_token, dist_token, seq), dim=0)
        # pos embedding
        seq = self.pos_embed(seq)
        seq = self.transformer(seq)

        cls_token = seq[0]
        dist_token = seq[1]

        prob = self.cls_head(cls_token)
        dist = self.dist_head(dist_token)

        if self.training:
            ret = {
                "pred": prob,
                "dist": dist
            }
        else:
            # during inference, return the average of both classifier predictions
            ret = (prob + dist) / 2
        return ret

