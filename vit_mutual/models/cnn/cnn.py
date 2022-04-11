from typing import Callable, List

import torch
import torch.nn as nn

from vit_mutual.models.layers import get_activation_fn


class CNN(nn.Module):
    def __init__(
        self,
        input_proj: nn.Module,
        base_block: Callable[[], nn.Module],
        embed_dim: int,
        num_layers: int,
        num_classes: int,
        activation: str = "relu",
        down_sample_layers: List[int] = list()
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_proj = input_proj
        self.layers = nn.ModuleList([base_block() for _ in range(num_layers)])
        self.bn = nn.BatchNorm2d(embed_dim)
        self.activation = get_activation_fn(activation)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(embed_dim, num_classes)

        self.downsample = nn.ModuleDict()
        for layer_id in range(num_layers):
            if layer_id in down_sample_layers:
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                layer = nn.Identity()
            self.downsample[f"{layer_id}"] = layer

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            x = self.downsample[f"{layer_id}"](x)
        x = self.activation(self.bn(x))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        pred = self.linear(x)
        return pred

