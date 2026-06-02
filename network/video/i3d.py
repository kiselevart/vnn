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


class AuxiliaryHead(nn.Module):
    """
    Auxiliary classification head (GoogLeNet / I3D paper).

    Attached at mixed_4b and mixed_4e during training (weight 0.3).
    Discarded at inference — call only when model.training.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((None, 4, 4))  # spatial downsample only
        self.conv = Unit3D(in_channels, 128)              # 1×1×1
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.dropout(x))


class I3D(nn.Module):
    """
    Single-stream I3D backbone.

    Args:
        num_classes: number of output classes
        in_channels: 3 for RGB, 2 for optical flow (u, v)
        dropout_prob: applied before the final linear layer (paper: 0.5)
        clip_len: input temporal length — controls early temporal strides
        width_mult: uniform channel scale factor (1.0 = full I3D ~12.5M/stream,
                    0.5 = small I3D ~3.1M/stream → ~6.2M two-stream)
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 dropout_prob: float = 0.5, clip_len: int = 16,
                 width_mult: float = 1.0):
        super().__init__()

        def _c(n: int) -> int:
            """Scale n by width_mult, round to nearest multiple of 8, min 8."""
            return max(8, round(int(n * width_mult) / 8) * 8)

        def _out(spec):
            """Output channels of an InceptionModule given its branch_spec."""
            return spec[0] + spec[2] + spec[4] + spec[5]

        t_stride = 1 if clip_len <= 32 else 2

        # --- Stem ---
        c64 = _c(64); c192 = _c(192)
        self.conv1a = Unit3D(in_channels, c64, kernel_size=(7, 7, 7),
                             stride=(t_stride, 2, 2), padding=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1))
        self.conv2b = Unit3D(c64, c64)
        self.conv2c = Unit3D(c64, c192, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                  padding=(0, 1, 1))

        # --- Mixed 3 ---                         b0       b1r      b1       b2r     b2      b3
        m3b = (_c(64),  _c(96),  _c(128), _c(16), _c(32),  _c(32))
        m3c = (_c(128), _c(128), _c(192), _c(32), _c(96),  _c(64))
        self.mixed_3b = InceptionModule(c192,        m3b)
        self.mixed_3c = InceptionModule(_out(m3b),   m3c)
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                  padding=(1, 1, 1))

        # --- Mixed 4 ---
        m4b = (_c(192), _c(96),  _c(208), _c(16), _c(48),  _c(64))
        m4c = (_c(160), _c(112), _c(224), _c(24), _c(64),  _c(64))
        m4d = (_c(128), _c(128), _c(256), _c(24), _c(64),  _c(64))
        m4e = (_c(112), _c(144), _c(288), _c(32), _c(64),  _c(64))
        m4f = (_c(256), _c(160), _c(320), _c(32), _c(128), _c(128))
        self.mixed_4b = InceptionModule(_out(m3c),   m4b)
        self.mixed_4c = InceptionModule(_out(m4b),   m4c)
        self.mixed_4d = InceptionModule(_out(m4c),   m4d)
        self.mixed_4e = InceptionModule(_out(m4d),   m4e)
        self.mixed_4f = InceptionModule(_out(m4e),   m4f)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # --- Mixed 5 ---
        m5b = (_c(256), _c(160), _c(320), _c(32), _c(128), _c(128))
        m5c = (_c(384), _c(192), _c(384), _c(48), _c(128), _c(128))
        self.mixed_5b = InceptionModule(_out(m4f),   m5b)
        self.mixed_5c = InceptionModule(_out(m5b),   m5c)

        # --- Head ---
        c_feat = _out(m5c)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(c_feat, num_classes)

        # Auxiliary classifiers attached after mixed_4b and mixed_4e
        self.aux1 = AuxiliaryHead(_out(m4b), num_classes)
        self.aux2 = AuxiliaryHead(_out(m4e), num_classes)

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

    def forward(self, x: torch.Tensor):
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
        aux1 = self.aux1(x) if self.training else None
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        aux2 = self.aux2(x) if self.training else None
        x = self.mixed_4f(x)
        x = self.pool4(x)

        # Mixed 5
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)

        # Head
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        main = self.fc(x)

        if self.training:
            return main, [aux1, aux2]
        return main

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
        width_mult: channel scale factor passed to both I3D streams
        clip_len: input temporal length
    """

    def __init__(self, num_classes: int, dropout_prob: float = 0.5,
                 clip_len: int = 16, width_mult: float = 1.0):
        super().__init__()
        self.rgb_net  = I3D(num_classes, in_channels=3, dropout_prob=dropout_prob,
                            clip_len=clip_len, width_mult=width_mult)
        self.flow_net = I3D(num_classes, in_channels=2, dropout_prob=dropout_prob,
                            clip_len=clip_len, width_mult=width_mult)

    def forward(self, x):
        rgb, flow = x
        rgb_out  = self.rgb_net(rgb)
        flow_out = self.flow_net(flow)
        if self.training:
            rgb_main,  rgb_aux  = rgb_out
            flow_main, flow_aux = flow_out
            # aux: list of tensors from both streams, to be weighted 0.3 each
            return rgb_main, flow_main, rgb_aux + flow_aux
        return rgb_out, flow_out

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


def SmallI3DTwoStream(num_classes: int, dropout_prob: float = 0.5,
                     clip_len: int = 16) -> I3DTwoStream:
    """
    Width-halved two-stream I3D (~6.2M params total vs ~25.1M for full I3D).
    All Inception branch channels scaled by 0.5, rounded to nearest 8.
    Auxiliary classifiers and training interface identical to I3DTwoStream.
    """
    return I3DTwoStream(num_classes, dropout_prob=dropout_prob,
                        clip_len=clip_len, width_mult=0.5)
