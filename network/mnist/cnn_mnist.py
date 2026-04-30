"""Tiny two-block CNN baseline for MNIST."""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """Two-block CNN baseline for MNIST.

    Block 1: Conv2d(1  → ch,   3×3) + BN + ReLU → MaxPool(2) → [ch,   14, 14]
    Block 2: Conv2d(ch → 2ch,  3×3) + BN + ReLU → MaxPool(2) → [2ch,   7,  7]
    GAP → FC(2ch → num_classes)

    With base_ch=8: ~1.4 K params.
    """

    def __init__(self, num_classes: int = 10, base_ch: int = 8):
        super().__init__()
        ch = base_ch
        self.block1 = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)
