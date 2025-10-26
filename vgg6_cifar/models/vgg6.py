
from __future__ import annotations
import torch
import torch.nn as nn
from .activations import get_activation

class VGG6(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3, act_name: str = "relu", use_bn: bool = True):
        super().__init__()
        act = lambda: get_activation(act_name)
        def block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=not use_bn)]
            if use_bn:
                layers += [nn.BatchNorm2d(out_c)]
            layers += [act()]
            return nn.Sequential(*layers)
        self.features = nn.Sequential(
            block(3, 64), block(64, 64), nn.MaxPool2d(2),
            block(64, 128), block(128, 128), nn.MaxPool2d(2),
            block(128, 256), block(256, 256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 256),
            act(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights(use_bn)
    def _init_weights(self, use_bn: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif use_bn and isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
