import os
from typing import Union, List

import torch
import torch.nn as nn

from vit_mutual.models import ModelWrapper


def load_pretrain_model(pretrain_fp: str, model: Union[nn.Module, ModelWrapper], lax_names: List[str] = list()):
    """
    Weights has lax_names will remains the same when size mismatch.
    """
    pretrain_fp = os.path.expanduser(os.path.expandvars(pretrain_fp))
    ckpt = torch.load(pretrain_fp, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]

    if isinstance(model, ModelWrapper):
        model = model.module

    # different category number for vit
    state_dict = model.state_dict()
    for name in lax_names:
        if (name in state_dict) and (ckpt[name].shape != state_dict[name].shape):
            ckpt[name] = state_dict[name]
    model.load_state_dict(ckpt, strict=True)


