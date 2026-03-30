"""
I3D: Inflated 3D ConvNets.
Carreira & Zisserman, CVPR 2017 — "Quo Vadis, Action Recognition?"

Architecture: GoogLeNet/Inception-v1 inflated to 3D with temporal-kernel bootstrapping.
All Inception module channel counts match Table S1 of the paper.

Temporal stride notes
---------------------
The paper uses 64-frame clips.  For shorter clips (≤ 32 frames) the early
temporal strides are set to 1 so the network doesn't collapse the temporal
dimension before the Inception blocks see it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit3D(nn.Module):
    """Conv3d + BN + ReLU building block."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size=(1, 1, 1), stride=(1, 1, 1),
                 padding=0, use_relu: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)
        self.use_relu = use_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = F.relu(x, inplace=True)
        return x


class InceptionModule(nn.Module):
    """
    Inflated Inception module (Table S1).

    branch_spec: (b0, b1_reduce, b1, b2_reduce, b2, b3)
      b0       — 1×1×1 branch output channels
      b1_reduce— bottleneck before 3×3×3 branch
      b1       — 3×3×3 branch output channels
      b2_reduce— bottleneck before second 3×3×3 branch
                 (replaces the 5×5 from vanilla Inception, as in the paper)
      b2       — second 3×3×3 branch output channels
      b3       — pool-projection branch output channels
    """

    def __init__(self, in_channels: int, branch_spec):
        super().__init__()
        b0, b1r, b1, b2r, b2, b3 = branch_spec

        self.branch0 = Unit3D(in_channels, b0)

        self.branch1 = nn.Sequential(
            Unit3D(in_channels, b1r),
            Unit3D(b1r, b1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.branch2 = nn.Sequential(
            Unit3D(in_channels, b2r),
            Unit3D(b2r, b2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            Unit3D(in_channels, b3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch0(x),
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
        ], dim=1)


class I3D(nn.Module):
    """
    Single-stream I3D backbone.

    Args:
        num_classes: number of output classes
        in_channels: 3 for RGB, 2 for optical flow (u, v)
        dropout_prob: applied before the final linear layer (paper: 0.5)
        clip_len: input temporal length — controls early temporal strides
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 dropout_prob: float = 0.5, clip_len: int = 16):
        super().__init__()

        # For short clips, keep temporal dimension alive through the stem.
        t_stride = 1 if clip_len <= 32 else 2

        # --- Stem ---
        # Conv3d_1a_7x7
        self.conv1a = Unit3D(in_channels, 64, kernel_size=(7, 7, 7),
                             stride=(t_stride, 2, 2), padding=(3, 3, 3))
        # MaxPool3d_2a_3x3 — temporal stride 1 for short clips
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1))
        # Conv3d_2b_1x1
        self.conv2b = Unit3D(64, 64)
        # Conv3d_2c_3x3
        self.conv2c = Unit3D(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # MaxPool3d_3a_3x3 — temporal stride 1 for short clips
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1))

        # --- Mixed 3 ---                in   b0   b1r  b1   b2r  b2   b3
        self.mixed_3b = InceptionModule(192, (64,  96,  128, 16,  32,  32))
        self.mixed_3c = InceptionModule(256, (128, 128, 192, 32,  96,  64))
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1))

        # --- Mixed 4 ---
        self.mixed_4b = InceptionModule(480, (192, 96,  208, 16,  48,  64))
        self.mixed_4c = InceptionModule(512, (160, 112, 224, 24,  64,  64))
        self.mixed_4d = InceptionModule(512, (128, 128, 256, 24,  64,  64))
        self.mixed_4e = InceptionModule(512, (112, 144, 288, 32,  64,  64))
        self.mixed_4f = InceptionModule(528, (256, 160, 320, 32,  128, 128))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # --- Mixed 5 ---
        self.mixed_5b = InceptionModule(832, (256, 160, 320, 32,  128, 128))
        self.mixed_5c = InceptionModule(832, (384, 192, 384, 48,  128, 128))
        # Mixed_5c output: 384 + 384 + 128 + 128 = 1024

        # --- Head ---
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.conv1a(x)
        x = self.pool1(x)
        x = self.conv2b(x)
        x = self.conv2c(x)
        x = self.pool2(x)

        # Mixed 3
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.pool3(x)

        # Mixed 4
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.pool4(x)

        # Mixed 5
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)

        # Head
        x = self.avg_pool(x)      # [B, 1024, 1, 1, 1]
        x = x.flatten(1)           # [B, 1024]
        x = self.dropout(x)
        return self.fc(x)          # [B, num_classes]

    def get_1x_lr_params(self):
        fc_ids = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in fc_ids:
                yield p

    def get_10x_lr_params(self):
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p


class I3DTwoStream(nn.Module):
    """
    Two-stream I3D: one RGB network + one optical flow network.

    forward() returns (rgb_logits, flow_logits) during training so the caller
    can compute per-stream losses.  At evaluation, average the two before
    taking argmax.

    Args:
        num_classes: number of output classes
        dropout_prob: per-stream dropout (paper: 0.5)
        clip_len: input temporal length
    """

    def __init__(self, num_classes: int, dropout_prob: float = 0.5,
                 clip_len: int = 16):
        super().__init__()
        self.rgb_net  = I3D(num_classes, in_channels=3, dropout_prob=dropout_prob,
                            clip_len=clip_len)
        self.flow_net = I3D(num_classes, in_channels=2, dropout_prob=dropout_prob,
                            clip_len=clip_len)

    def forward(self, x):
        rgb, flow = x
        return self.rgb_net(rgb), self.flow_net(flow)

    def get_1x_lr_params(self):
        fc_ids = {id(p)
                  for net in (self.rgb_net, self.flow_net)
                  for p in net.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in fc_ids:
                yield p

    def get_10x_lr_params(self):
        for net in (self.rgb_net, self.flow_net):
            for p in net.fc.parameters():
                if p.requires_grad:
                    yield p
