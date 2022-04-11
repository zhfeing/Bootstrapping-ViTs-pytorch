from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn

from cv_lib.utils import MidExtractor


class MutualModel(nn.Module):
    def __init__(
        self,
        models: nn.ModuleDict,
        extractors: Dict[str, MidExtractor],
    ):
        super().__init__()
        self.models = models
        self.extractors = extractors

    def forward(self, x: torch.Tensor):
        preds = OrderedDict()
        mid_features = OrderedDict()
        for name, model in self.models.items():
            preds[name] = model(x)
            mid_features[name] = self.extractors[name].features
        ret = {
            "preds": preds,
            "mid_features": mid_features
        }
        return ret

