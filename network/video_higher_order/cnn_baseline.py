"""
cnn_baseline.py — Plain 3D ResNet baseline for video classification.

ResNet3D with BasicBlock3D targeting ~25M parameters for fair comparison
with VNN ablation models (sym25, gen25, quad25).

Architecture:
  Stem:   3 → 64,  Conv3d(1,2,2 stride)
  Stage1: 64,  2 blocks, stride 1
  Stage2: 128, 2 blocks, stride 2 (spatial only)
  Stage3: 256, 2 blocks, stride 2 (spatial only)
  Stage4: 416, 2 blocks, stride 2 (spatial only)
  Head:   GAP → FC(num_classes)

Total: ~25.3M parameters
"""

import torch
import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=(1, stride, stride), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1,
                    stride=(1, stride, stride), bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


def _make_stage(in_channels, out_channels, num_blocks, stride):
    layers = [BasicBlock3D(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class CNNBaseline3D(nn.Module):
    """Plain 3D ResNet baseline (~25.3M params). RGB-only input."""

    def __init__(self, num_classes=101):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _make_stage(64,  64,  num_blocks=2, stride=1)
        self.stage2 = _make_stage(64,  128, num_blocks=2, stride=2)
        self.stage3 = _make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = _make_stage(256, 416, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(416, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
