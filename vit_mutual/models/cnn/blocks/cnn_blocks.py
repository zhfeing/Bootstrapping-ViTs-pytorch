from typing import Callable, Any, Dict

import torch
import torch.nn as nn

import vit_mutual.models.layers as layers


############################################################################################
# CNN Block for replacing MHSA+MLP

class CNNBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        conv_block: nn.Module,
        mlp_block: nn.Module,
        norm: Callable[[Dict[str, Any]], nn.Module],
        dropout: float = None,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.cnn_block = conv_block
        self.mlp_block = mlp_block
        self.dropout1 = layers.get_dropout(dropout)
        self.dropout2 = layers.get_dropout(dropout)
        self.norm1 = norm(num_features=embed_dim)
        self.norm2 = norm(num_features=embed_dim)
        self.identity1 = nn.Identity()
        self.identity2 = nn.Identity()
        self.pre_norm = pre_norm

    def pre_forward(self, x: torch.Tensor):
        y = self.norm1(x)
        y = self.cnn_block(y)
        self.identity1(x + y)
        x = x + self.dropout1(y)

        y = self.norm2(x)
        y = self.mlp_block(y)
        self.identity2(x + y)
        x = x + self.dropout1(y)
        return x

    def post_forward(self, x: torch.Tensor):
        y = self.cnn_block(x)
        y = self.norm1(y)
        self.identity1(x + y)
        x = x + self.dropout1(y)

        y = self.mlp_block(x)
        y = self.norm2(y)
        self.identity2(x + y)
        x = x + self.dropout1(y)
        return x

    def forward(self, x: torch.Tensor):
        if self.pre_norm:
            forward_fn = self.pre_forward
        else:
            forward_fn = self.post_forward
        return forward_fn(x)


__REGISTERED_CNN_BLOCK__ = {
    "cnn_block": CNNBlock,
}


def get_cnn_block(
    name: int,
    embed_dim: int,
    conv_block: Callable[[], nn.Module],
    mlp_block: Callable[[], nn.Module],
    norm: Callable[[Dict[str, Any]], nn.Module],
    dropout: float = None,
    pre_norm: bool = True,
):
    cnn = __REGISTERED_CNN_BLOCK__[name](
        embed_dim=embed_dim,
        conv_block=conv_block(),
        mlp_block=mlp_block(),
        norm=norm,
        dropout=dropout,
        pre_norm=pre_norm
    )
    return cnn

