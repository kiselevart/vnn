"""Tiny two-block quadratic VNN for MNIST."""

import torch
import torch.nn as nn


class VolterraBlock2D(nn.Module):
    """Single quadratic Volterra block for 2D images.

    out = BN(conv_lin(x)) + gate · BN(conv_left(x) ⊙ conv_right(x))

    gate is per-channel, initialised at 1e-4 so the block starts near-linear
    and gradually opens the quadratic interaction during training.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_lin   = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn_lin     = nn.BatchNorm2d(out_ch)
        self.conv_left  = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.conv_right = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn_quad    = nn.BatchNorm2d(out_ch)
        self.gate       = nn.Parameter(torch.ones(out_ch) * 1e-4)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = x.clamp(-50.0, 50.0)
        lin  = self.bn_lin(self.conv_lin(x))
        quad = self.bn_quad(
            (self.conv_left(x) * self.conv_right(x)).clamp(-50.0, 50.0)
        )
        return lin + self.gate.view(1, -1, 1, 1) * quad


class TinyVNN(nn.Module):
    """Two-block quadratic VNN for MNIST.

    Block 1: VolterraBlock2D(1  → ch)  + MaxPool(2) → [ch,  14, 14]
    Block 2: VolterraBlock2D(ch → 2ch) + MaxPool(2) → [2ch,  7,  7]
    GAP → FC(2ch → num_classes)

    With base_ch=8: ~4 K params (3 convs per block: lin + left + right).
    """

    def __init__(self, num_classes: int = 10, base_ch: int = 8):
        super().__init__()
        ch = base_ch
        self.block1     = VolterraBlock2D(1, ch)
        self.pool1      = nn.MaxPool2d(2)
        self.block2     = VolterraBlock2D(ch, ch * 2)
        self.pool2      = nn.MaxPool2d(2)
        self.classifier = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = x.mean(dim=(2, 3))
        return self.classifier(x)
