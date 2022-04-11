from copy import deepcopy
from typing import Dict, Any

import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import LayerNorm


class Norm_fn:
    def __init__(self, norm_cfg: Dict[str, Any]):
        self.norm_name = norm_cfg["name"]
        self.norm_cfg = norm_cfg

    def _bn_call(self, **runtime_kwargs) -> nn.Module:
        cfg = deepcopy(self.norm_cfg)
        cfg.update(runtime_kwargs)
        return BatchNorm2d(
            num_features=cfg["num_features"],
            eps=cfg.pop("eps", 1e-5),
            momentum=cfg.pop("momentum", 0.1),
            affine=cfg.pop("affine", True)
        )

    def _ln_call(self, **runtime_kwargs) -> nn.Module:
        cfg = deepcopy(self.norm_cfg)
        cfg.update(runtime_kwargs)
        return LayerNorm(
            normalized_shape=cfg["normalized_shape"],
            eps=cfg.pop("eps", 1e-5),
            elementwise_affine=cfg.pop("momentum", True),
        )

    def __call__(self, **runtime_kwargs) -> nn.Module:
        fn = {
            "bn": self._bn_call,
            "ln": self._ln_call
        }
        return fn[self.norm_name](**runtime_kwargs)
