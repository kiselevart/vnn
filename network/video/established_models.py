"""
Established video architectures for baseline comparison.

All models use torchvision implementations (no custom code) with the output
FC layer replaced for the target number of classes.  They accept inputs of
shape [B, 3, T, H, W] with any T ≥ 8; global average pooling makes them
clip-length agnostic.

Models
------
  R2Plus1DNet   — R(2+1)D-18 (31.5M params)
                  Factorized (spatial 1×k×k then temporal T×1×1) residual net.
                  Tran et al. 2018 report ~74.5% UCF101 from scratch.

  R3DNet        — R3D-18 (33.4M params)
                  Full 3D ResNet-18.  Ablation partner for R(2+1)D:
                  same capacity, no factorization.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.models.video as vm


class R2Plus1DNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = vm.r2plus1d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.model.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.model.fc.parameters():
            if p.requires_grad:
                yield p


class R3DNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = vm.r3d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.model.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.model.fc.parameters():
            if p.requires_grad:
                yield p


class ResNet50FrameAvg(nn.Module):
    """
    2D ResNet-50 with per-frame inference and temporal average pooling.

    Processes each frame of a clip independently through ResNet-50, then
    averages the resulting feature vectors across time before classification.
    This is the simplest possible temporal baseline: no 3D convolutions,
    no motion information — pure spatial features averaged over time.

    Input:  [B, 3, T, H, W]
    Output: [B, num_classes]
    """

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = tvm.resnet50(weights=None)
        self.feature_dim = backbone.fc.in_features               # 2048
        # Drop the FC; keep conv stem + layers + avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.backbone(frames).flatten(1)             # [B*T, 2048]
        feats = feats.view(B, T, self.feature_dim).mean(1)  # [B, 2048]
        return self.fc(feats)

    def get_1x_lr_params(self):
        fc_ids = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in fc_ids:
                yield p

    def get_10x_lr_params(self):
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p
