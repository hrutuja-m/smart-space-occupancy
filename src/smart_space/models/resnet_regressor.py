"""ResNet18 backbone used two ways:

* ``build_regressor()``          -> ResNet18 with a scalar head (direct count regression)
* ``build_feature_extractor()``  -> ResNet18 minus the FC layer (512-d embeddings
                                    consumed by the XGBoost head)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_regressor(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


def build_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    backbone = nn.Sequential(*list(net.children())[:-1])  # drop FC -> (B, 512, 1, 1)
    backbone.eval()
    return backbone


@torch.no_grad()
def extract_features(backbone: nn.Module, loader, device: str):
    """Run a dataloader through the backbone -> (X: (N,512), y: (N,))."""
    import numpy as np

    feats, targets = [], []
    backbone.to(device).eval()
    for x, y in loader:
        out = backbone(x.to(device)).flatten(1)
        feats.append(out.cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(feats), np.concatenate(targets)
