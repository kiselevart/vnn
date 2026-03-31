"""
cnn_baseline.py — R3D-18 baseline for video classification.

Exact replica of torchvision's r3d_18 architecture (no pretrained weights).
Full 3D ResNet-18: BasicBlock with 3×3×3 convolutions throughout.

Architecture (matches torchvision.models.video.r3d_18):
  Stem:   Conv3d(3→64, k=(3,7,7), stride=(1,2,2), pad=(1,3,3)) → BN → ReLU
          MaxPool3d(k=(1,3,3), stride=(1,2,2), pad=(0,1,1))
  Layer1: 64,  2 blocks, stride 1
  Layer2: 128, 2 blocks, stride (1,2,2)
  Layer3: 256, 2 blocks, stride (1,2,2)
  Layer4: 512, 2 blocks, stride (1,2,2)
  Head:   AdaptiveAvgPool3d(1) → FC(num_classes)

Total: ~33.4M parameters
"""

import torch.nn as nn


class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=(1, stride, stride), padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1,
                    stride=(1, stride, stride), bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _make_layer(in_channels, out_channels, num_blocks, stride):
    layers = [BasicBlock3D(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock3D(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class CNNBaseline3D(nn.Module):
    """R3D-18: full 3D ResNet-18 (~33.4M params). RGB-only single-stream."""

    def __init__(self, num_classes=101):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                      stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.layer1 = _make_layer(64,  64,  num_blocks=2, stride=1)
        self.layer2 = _make_layer(64,  128, num_blocks=2, stride=2)
        self.layer3 = _make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = _make_layer(256, 512, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).flatten(1))

    def get_1x_lr_params(self):
        fc_ids = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in fc_ids:
                yield p

    def get_10x_lr_params(self):
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p
