import math
import logging
from typing import Any, Dict, Callable, List, Tuple
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_lib.utils import MidExtractor

from vit_mutual.models.vision_transformers import ViT
from vit_mutual.models.transformer.transformer import MLP, MultiHeadSelfAttention
from vit_mutual.models.cnn.blocks import CNNBlock
from vit_mutual.models.cnn.blocks.conv import conv_2d, Conv_2d
from vit_mutual.models.layers import get_activation_fn, Norm_fn
from vit_mutual.models.cnn import get_input_proj


class SharedConv(nn.Module):
    def __init__(self, mhsa: MultiHeadSelfAttention):
        super().__init__()
        self.mhsa = mhsa

        self.kernel_size = math.ceil(math.sqrt(mhsa.num_heads))
        self.phi: torch.Tensor = None
        self.last_shape: Tuple[int, int] = None

    def get_phi(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        phi = Conv_2d.get_phi(
            shape=shape,
            device=device,
            kernel_size=(self.kernel_size, self.kernel_size),
            flatten=False
        )
        phi = phi[:self.mhsa.num_heads]
        phi = phi.permute(1, 0, 2).flatten(1)
        return phi

    def forward(self, x: torch.Tensor):
        H = self.mhsa.num_heads
        d_k = self.mhsa.head_dim

        # [Hxd_k, d] -> [H, d_k, d]
        weight_v = self.mhsa.get_weight_v().unflatten(0, (H, d_k))
        # [d, Hxd_k] -> [d, H, d_k] -> [H, d, d_k]
        weight_o = self.mhsa.get_weight_o().unflatten(1, (H, d_k)).transpose(0, 1)
        weights = torch.bmm(weight_o, weight_v)

        shape = x.shape[2:]
        if self.last_shape == shape and self.phi is not None:
            phi = self.phi
        else:
            phi = self.get_phi(shape, x.device)
            self.last_shape = shape
            self.phi = phi
        return conv_2d(x, phi=self.phi, weights=weights, bias=self.mhsa.get_bias_o())


class SharedLinearProjection(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias

    def forward(self, x: torch.Tensor):
        # reshape as [out_dim, in_dim, 1, 1]
        return F.conv2d(x, self.weight[..., None, None], self.bias)


class SharedMLP(nn.Module):
    def __init__(self, mlp: MLP):
        super().__init__()
        self.linear1 = SharedLinearProjection(mlp.linear1)
        self.linear2 = SharedLinearProjection(mlp.linear2)
        self.dropout = deepcopy(mlp.dropout)
        self.activation = deepcopy(mlp.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x


class MutualCNN(nn.Module):
    def __init__(
        self,
        input_proj: nn.Module,
        conv_blocks: nn.ModuleList,
        mlp_blocks: nn.ModuleList,
        embed_dim: int,
        num_classes: int,
        norm_fn: Callable[[Dict[str, Any]], nn.Module],
        activation: str = "relu",
        dropout: float = None,
        down_sample_layers: List[int] = list(),
        pre_norm: bool = True,
    ):
        super().__init__()
        self.input_proj = input_proj
        layers = nn.ModuleList()
        for b1, b2 in zip(conv_blocks, mlp_blocks):
            block = CNNBlock(
                embed_dim=embed_dim,
                conv_block=b1,
                mlp_block=b2,
                norm=norm_fn,
                dropout=dropout,
                pre_norm=pre_norm
            )
            layers.append(block)
        self.layers = layers
        self.bn = nn.BatchNorm2d(embed_dim)
        self.activation = get_activation_fn(activation)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(embed_dim, num_classes)

        self.downsample = nn.ModuleDict()
        for layer_id in range(len(self.layers)):
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


class JointModel(nn.Module):
    def __init__(
        self,
        vit: ViT,
        embed_dim: int,
        input_proj_cfg: Dict[str, Any],
        norm_cfg: Dict[str, Any],
        activation: str = "relu",
        dropout: float = None,
        extract_cnn: List[str] = list(),
        extract_vit: List[str] = list(),
        down_sample_layers: List[int] = list(),
        cnn_pre_norm: bool = True,
        **kwargs
    ):
        super().__init__()
        # input embedding layer
        input_proj = get_input_proj(embed_dim, input_proj_cfg)
        # share layers
        mhsa = [SharedConv(mhsa) for mhsa in vit.get_mhsa()]
        mlp = [SharedMLP(mlp) for mlp in vit.get_mlp()]
        norm_fn = Norm_fn(norm_cfg)

        cnn = MutualCNN(
            input_proj=input_proj,
            conv_blocks=nn.ModuleList(mhsa),
            mlp_blocks=nn.ModuleList(mlp),
            embed_dim=embed_dim,
            num_classes=vit.num_classes,
            norm_fn=norm_fn,
            activation=activation,
            dropout=dropout,
            down_sample_layers=down_sample_layers,
            pre_norm=cnn_pre_norm
        )
        self.models = nn.ModuleDict()
        self.extractors: Dict[str, MidExtractor] = OrderedDict()
        self.models["cnn"] = cnn
        self.extractors["cnn"] = MidExtractor(self.models["cnn"], extract_cnn)
        self.models["vit"] = vit
        self.extractors["vit"] = MidExtractor(self.models["vit"], extract_vit)

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


