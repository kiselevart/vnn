"""
Established video architectures for baseline comparison.

All models accept inputs of shape [B, 3, T, H, W]; global average pooling
makes them clip-length agnostic.

Models
------
  R2Plus1DNet   — R(2+1)D-18 (31.5M params)
  R3DNet        — R3D-18 (33.4M params)
  ResNet50FrameAvg — 2D ResNet-50 with per-frame inference + temporal avg

  SmallR3D      — R3D-18 style, channels [32,64,128,192] (~5.8M params)
  SmallR2Plus1D — R(2+1)D-18 style, same channels (~5.8M params)
                  Both are parameter-budget matches for the LVN/ortho models.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.models.video as vm

# ---------------------------------------------------------------------------
# Shared building blocks for SmallR3D / SmallR2Plus1D
# ---------------------------------------------------------------------------

_SMALL_CHANNELS = (32, 64, 128, 192)
_SMALL_BLOCKS   = (2, 2, 2, 2)


def _make_layer(block_cls, in_ch, out_ch, num_blocks, stride):
    layers = [block_cls(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(block_cls(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


class _Block3D(nn.Module):
    """Standard 3×3×3 residual block (R3D style)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + (self.shortcut(x) if self.shortcut is not None else x))


class _Block2Plus1D(nn.Module):
    """Factorized (spatial 1×3×3 → temporal 3×1×1) residual block (R(2+1)D style).

    Midplane count follows the formula from Tran et al. 2018 to keep param count
    approximately equal to a full 3×3×3 conv at the same in/out channels.
    """

    @staticmethod
    def _mid(in_ch, out_ch):
        return max(1, (in_ch * out_ch * 27) // (in_ch * 9 + 3 * out_ch))

    def _factorized_conv(self, in_ch, out_ch, stride):
        mid = self._mid(in_ch, out_ch)
        return nn.Sequential(
            nn.Conv3d(in_ch, mid, (1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, out_ch, (3, 1, 1), stride=(stride, 1, 1), padding=(1, 0, 0), bias=False),
        )

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = self._factorized_conv(in_ch, out_ch, stride)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = self._factorized_conv(out_ch, out_ch, 1)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + (self.shortcut(x) if self.shortcut is not None else x))


class _SmallVideoResNet(nn.Module):
    """Shared backbone for SmallR3D and SmallR2Plus1D."""

    def __init__(self, block_cls, num_classes,
                 channels=_SMALL_CHANNELS, blocks=_SMALL_BLOCKS):
        super().__init__()
        ch = channels
        self.stem = nn.Sequential(
            nn.Conv3d(3, ch[0], (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(block_cls, ch[0], ch[0], blocks[0], stride=1)
        self.layer2 = _make_layer(block_cls, ch[0], ch[1], blocks[1], stride=2)
        self.layer3 = _make_layer(block_cls, ch[1], ch[2], blocks[2], stride=2)
        self.layer4 = _make_layer(block_cls, ch[2], ch[3], blocks[3], stride=2)
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.fc     = nn.Linear(ch[3], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).flatten(1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p


class SmallR3D(_SmallVideoResNet):
    """R3D-18 style with channels [32,64,128,192] — ~5.8M params.

    Parameter-budget match for the LVN/ortho fusion models (6.2M).
    Single-stream RGB input [B, 3, T, H, W].
    """

    def __init__(self, num_classes: int):
        super().__init__(_Block3D, num_classes)


class SmallR2Plus1D(_SmallVideoResNet):
    """R(2+1)D-18 style with channels [32,64,128,192] — ~5.8M params.

    Same channel/depth config as SmallR3D; differs only in the factorized
    (spatial 1×3×3 then temporal 3×1×1) convolution blocks.
    Single-stream RGB input [B, 3, T, H, W].
    """

    def __init__(self, num_classes: int):
        super().__init__(_Block2Plus1D, num_classes)


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
