from typing import Any, Dict

import torch.nn as nn

import cv_lib.classification.models as cv_models
from .vision_transformers import get_vit, get_deit
from .cnn import get_cnn
from timm.models.efficientnet import efficientnet_b2


cv_models.register_model("efficientnet_b2", efficientnet_b2)


__REGISTERED_MODELS__ = {
    "vit": get_vit,
    "deit": get_deit,
    "cnn": get_cnn,
    "official_models": cv_models.get_model
}


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, dict):
            return output
        ret = {
            "pred": output
        }
        return ret


def get_model(model_cfg: Dict[str, Any], num_classes: int, with_wrapper: bool = True) -> nn.Module:
    model = __REGISTERED_MODELS__[model_cfg["name"]](model_cfg, num_classes)
    if with_wrapper:
        model = ModelWrapper(model)
    return model
