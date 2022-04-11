import logging
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Dict, Union

import torch
import torch.nn as nn

from cv_lib.config_parsing import get_cfg
from cv_lib.utils import MidExtractor

from vit_mutual.models import get_model
from .base_mutual_model import MutualModel
from .joint_model import JointModel


def get_base_mutual_model(model_cfg: Dict[str, Any], num_classes: int) -> MutualModel:
    logger = logging.getLogger("get_base_mutual_model")
    names = sorted(model_cfg.keys())
    models = nn.ModuleDict()
    extractors = OrderedDict()
    for name in names:
        cfg = model_cfg[name]
        m_cfg = get_cfg(cfg["cfg_path"])["model"]
        model = get_model(m_cfg, num_classes, False)
        logger.info("Built submodel: %s", name)
        # load from ckpt
        ckpt_path = cfg.get("ckpt", None)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            logger.info("Loaded ckpt for submodel: %s from dir: %s", name, ckpt_path)
        models[name] = model

        extractor = MidExtractor(model, cfg["extract_layers"])
        extractors[name] = extractor

    return MutualModel(models, extractors)


def get_joint_model(model_cfg: Dict[str, Any], num_classes: int) -> JointModel:
    # get vit base model
    vit_cfg = model_cfg["vit"]
    cnn_cfg = model_cfg["cnn"]

    vit = get_model(
        model_cfg=vit_cfg,
        num_classes=num_classes,
        with_wrapper=False
    )

    joint_model = JointModel(
        vit=vit,
        input_proj_cfg=cnn_cfg["input_proj"],
        norm_cfg=cnn_cfg["norm"],
        bias=True,
        extract_cnn=model_cfg["extract_layers_cnn"],
        extract_vit=model_cfg["extract_layers_vit"],
        down_sample_layers=cnn_cfg.get("down_sample_layers", list()),
        cnn_pre_norm=cnn_cfg["pre_norm"],
        **vit_cfg["transformer"],
    )
    return joint_model


__REGISTERED_MUTUAL_MODEL__ = {
    "base_mutual": get_base_mutual_model,
    "joint_model": get_joint_model
}


def get_mutual_model(mutual_model_cfg: Dict[str, Any], num_classes: int) -> Union[MutualModel, JointModel]:
    mutual_model_cfg = deepcopy(mutual_model_cfg)
    name = mutual_model_cfg.pop("name")
    model = __REGISTERED_MUTUAL_MODEL__[name](mutual_model_cfg, num_classes)
    return model

