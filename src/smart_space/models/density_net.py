"""CSRNet-lite: a compact dilated-convolution density estimator.

Architecture (follows CSRNet, Li et al. CVPR 2018, trimmed for small data):
  * Frontend  : first 10 conv layers of VGG16-BN (ImageNet pretrained), output stride 8.
  * Backend   : 4 dilated 3x3 convs (dilation 2) that enlarge the receptive field
                without further downsampling — the key CSRNet idea.
  * Head      : 1x1 conv -> single-channel density map, ReLU to keep mass >= 0.

Predicted count = sum over the density map. Training loss is MSE on the map.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CSRNetLite(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = models.vgg16_bn(weights=weights)
        # features[:33] = up to (and including) the 10th conv block of vgg16_bn,
        # giving an output stride of 8.
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])

        def dilated(inc: int, outc: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inc, outc, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
            )

        self.backend = nn.Sequential(
            dilated(512, 256),
            dilated(256, 128),
            dilated(128, 64),
            dilated(64, 64),
        )
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._init_backend()

    def _init_backend(self) -> None:
        for m in [*self.backend.modules(), self.output_layer]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return torch.relu(x)

    @torch.no_grad()
    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (B,) tensor of predicted counts (density-map integrals)."""
        return self.forward(x).sum(dim=(1, 2, 3))
