"""
LVN backbone and fusion models using standard Volterra interactions.

After removing the signed-Gaussian interaction experiments, LVNBackbone
and LVNHead are thin wrappers around the same VolterraBlock3D / MultiKernelBlock3D
primitives used by Backbone4Block.  The distinction from vnn_fusion_ho is the
named constructor entry point and the cubic_mode/use_cubic defaults.
"""

import torch
import torch.nn as nn

from .volterra_blocks import (
    VolterraBlock3D, MultiKernelBlock3D, ClassifierHead,
)


class LVNBackbone(nn.Module):
    def __init__(self, num_ch: int = 3, cubic_mode: str = 'symmetric', use_cubic: bool = True):
        super().__init__()
        self.block1 = MultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2,
        )
        self.block2 = VolterraBlock3D(24, 32, Q=4, stride=2, use_shortcut=True)
        self.block3 = VolterraBlock3D(32, 64, Q=4, Qc=2,
                                      use_cubic=use_cubic, cubic_mode=cubic_mode,
                                      use_shortcut=True)
        self.block4 = VolterraBlock3D(64, 96, Q=4, Qc=2, stride=2,
                                      use_cubic=use_cubic, cubic_mode=cubic_mode,
                                      use_shortcut=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class LVNHead(nn.Module):
    def __init__(self, num_classes: int, num_ch: int = 96,
                 cubic_mode: str = 'symmetric', use_cubic: bool = True,
                 clip_len: int = 16):
        super().__init__()
        self.block = VolterraBlock3D(num_ch, 256, Q=2, Qc=2, stride=2,
                                     use_cubic=use_cubic, cubic_mode=cubic_mode,
                                     use_shortcut=True, gate_quadratic=True)
        fc_features = 256 * (clip_len // 16) * 7 * 7
        self.classifier = ClassifierHead(fc_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block(x))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.classifier.fc.parameters():
            if p.requires_grad:
                yield p


class LVNRgb(nn.Module):
    def __init__(self, num_classes: int, cubic_mode: str = 'symmetric',
                 use_cubic: bool = True, clip_len: int = 16):
        super().__init__()
        self.backbone = LVNBackbone(num_ch=3, cubic_mode=cubic_mode, use_cubic=use_cubic)
        self.head     = LVNHead(num_classes=num_classes, num_ch=96,
                                cubic_mode=cubic_mode, use_cubic=use_cubic,
                                clip_len=clip_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.head.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.head.classifier.fc.parameters():
            if p.requires_grad:
                yield p


class LVNFusion(nn.Module):
    """Two-stream fusion with standard Volterra cross-stream interaction."""

    def __init__(self, num_classes: int, cubic_mode: str = 'symmetric',
                 use_cubic: bool = True, clip_len: int = 16):
        super().__init__()
        self.model_rgb = LVNBackbone(num_ch=3, cubic_mode=cubic_mode, use_cubic=use_cubic)
        self.model_of  = LVNBackbone(num_ch=2, cubic_mode=cubic_mode, use_cubic=use_cubic)
        self.cross_bn  = nn.BatchNorm3d(96)
        self.head      = LVNHead(num_classes=num_classes, num_ch=288,
                                 cubic_mode=cubic_mode, use_cubic=use_cubic,
                                 clip_len=clip_len)
        self.cross_abs_max = 0.0

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        cross   = torch.clamp(self.cross_bn(out_rgb * out_of), -50.0, 50.0)
        with torch.no_grad():
            self.cross_abs_max = cross.abs().max().item()
        return self.head(torch.cat((out_rgb, out_of, cross), dim=1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.head.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.head.classifier.fc.parameters():
            if p.requires_grad:
                yield p


def lvn_rgb_signed(num_classes: int, clip_len: int = 16) -> LVNRgb:
    return LVNRgb(num_classes, clip_len=clip_len)

def lvn_fusion_signed(num_classes: int, clip_len: int = 16) -> LVNFusion:
    return LVNFusion(num_classes, clip_len=clip_len)
