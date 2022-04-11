from functools import partial
from typing import Dict, Any

from .cnn import CNN
from .blocks import get_cnn_block, get_conv_block, get_mlp, get_input_proj
from vit_mutual.models.layers import Norm_fn


def get_cnn(cnn_cfg: Dict[str, Any], num_classes: int) -> CNN:
    pre_norm = cnn_cfg.get("pre_norm", True)

    embed_dim = cnn_cfg["embed_dim"]
    activation = cnn_cfg["activation"]
    norm_fn = Norm_fn(cnn_cfg["norm"])
    conv_fn = partial(
        get_conv_block,
        embed_dim=embed_dim,
        activation=activation,
        norm=norm_fn,
        conv_cfg=cnn_cfg["conv_block"]
    )
    mlp_fn = partial(
        get_mlp,
        embed_dim=embed_dim,
        activation=activation,
        mlp_cfg=cnn_cfg["mlp_block"]
    )
    cnn_block_fn = partial(
        get_cnn_block,
        name=cnn_cfg["cnn_name"],
        embed_dim=embed_dim,
        conv_block=conv_fn,
        mlp_block=mlp_fn,
        norm=norm_fn,
        dropout=cnn_cfg.get("dropout", None),
        pre_norm=pre_norm
    )
    input_proj = get_input_proj(
        embed_dim=embed_dim,
        proj_cfg=cnn_cfg["input_proj"]
    )
    cnn = CNN(
        input_proj=input_proj,
        base_block=cnn_block_fn,
        embed_dim=embed_dim,
        num_layers=cnn_cfg["num_layers"],
        num_classes=num_classes,
        activation=activation,
        down_sample_layers=cnn_cfg.get("down_sample_layers", list())
    )
    return cnn
