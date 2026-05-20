"""
Legacy VNN architecture — matches the original Volterra-Neural-Networks repo.

Use this for ablations that need to compare against the original paper results.

Differences from VNNFusionHO (current higher-order model):
  - No factor or output clamping anywhere (original was numerically raw)
  - No quadratic gate (quadratic branch contributes fully from epoch 0)
  - No residual shortcuts in backbone blocks
  - No cubic path
  - Block 1 uses (3,3,3)+(3,3,3)+(1,1,1) kernels — original naming was _5/_3/_1
    but all three kernel_size args were literally (3,3,3)/(3,3,3)/(1,1,1)
  - Two-stream fusion is plain cat(rgb, flow) → 192ch, no cross-stream product

Q is configurable (original hardcoded Q=4 for backbone, Q=2 for fusion head).
Pass --Q to vary backbone rank for the Q-sweep ablation.
"""

import torch
import torch.nn as nn

from .volterra_blocks import init_vnn_weights


def _quad_unclamped(x_conv, Q, nch_out):
    """Quadratic Volterra interaction without any clamping — matches original."""
    mid = Q * nch_out
    product = x_conv[:, :mid] * x_conv[:, mid:]          # [B, Q*C, T, H, W]
    return product.view(product.shape[0], Q, nch_out, *product.shape[2:]).sum(dim=1)


class LegacyBlock3D(nn.Module):
    """Single-kernel 3D Volterra block — no gate, no shortcut, no clamp, no cubic."""

    def __init__(self, in_ch, out_ch, Q=4, stride=1, kernel_size=3):
        super().__init__()
        self.Q = Q
        self.out_ch = out_ch
        ks = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        pad = (ks[0] // 2, ks[1] // 2, ks[2] // 2)

        self.conv_lin  = nn.Conv3d(in_ch, out_ch,         ks, padding=pad)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = nn.Conv3d(in_ch, 2 * Q * out_ch, ks, padding=pad)
        self.bn_quad   = nn.BatchNorm3d(out_ch)
        self.pool      = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        lin  = self.bn_lin(self.conv_lin(x))
        quad = self.bn_quad(_quad_unclamped(self.conv_quad(x), self.Q, self.out_ch))
        return self.pool(lin + quad)


class LegacyMultiKernelBlock3D(nn.Module):
    """Multi-kernel block matching original block 1.

    Kernels: (3,3,3) + (3,3,3) + (1,1,1), 8ch each → 24ch total.
    The original named these _5/_3/_1 but the actual kernel_size values were
    (3,3,3), (3,3,3), (1,1,1) — confirmed by reading the source.
    """

    KERNELS = [(3, 3, 3), (3, 3, 3), (1, 1, 1)]

    def __init__(self, in_ch, ch_per_kernel=8, Q=4, stride=2):
        super().__init__()
        self.Q           = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch      = ch_per_kernel * len(self.KERNELS)

        self.lin_convs = nn.ModuleList([
            nn.Conv3d(in_ch, ch_per_kernel, ks, padding=(ks[0] // 2, ks[1] // 2, ks[2] // 2))
            for ks in self.KERNELS
        ])
        self.bn_lin = nn.BatchNorm3d(self.out_ch)

        self.quad_convs = nn.ModuleList([
            nn.Conv3d(in_ch, 2 * Q * ch_per_kernel, ks, padding=(ks[0] // 2, ks[1] // 2, ks[2] // 2))
            for ks in self.KERNELS
        ])
        self.bn_quad = nn.BatchNorm3d(self.out_ch)

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        lin  = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))
        quad = self.bn_quad(torch.cat(
            [_quad_unclamped(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs],
            dim=1,
        ))
        return self.pool(lin + quad)


class LegacyBackbone4Block(nn.Module):
    """4-block backbone matching original vnn_rgb_of_highQ.VNN.

    Channels: 24 → 32 → 64 → 96.
    Q is uniform across all blocks (original hardcoded Q=4 everywhere).
    """

    def __init__(self, num_ch=3, Q=4):
        super().__init__()
        self.block1 = LegacyMultiKernelBlock3D(num_ch, ch_per_kernel=8, Q=Q, stride=2)
        self.block2 = LegacyBlock3D(24, 32, Q=Q, stride=2)
        self.block3 = LegacyBlock3D(32, 64, Q=Q, stride=1)
        self.block4 = LegacyBlock3D(64, 96, Q=Q, stride=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class LegacyFusionHead(nn.Module):
    """Fusion head matching original vnn_fusion_highQ.VNN_F.

    Single Volterra block → pool → dropout → FC.
    No gate, no shortcut, no cubic, no clamping.

    Args:
        num_classes: Output classes.
        num_ch: Input channels (192 for two-stream cat, 96 for RGB-only).
        Q: Quadratic rank (original used Q=2 here).
        clip_len: Clip length; determines FC input size.
    """

    def __init__(self, num_classes, num_ch=192, Q=2, clip_len=16):
        super().__init__()
        self.Q      = Q
        self.out_ch = 256

        self.conv_lin  = nn.Conv3d(num_ch, 256,           (3, 3, 3), padding=(1, 1, 1))
        self.bn_lin    = nn.BatchNorm3d(256)
        self.conv_quad = nn.Conv3d(num_ch, 2 * Q * 256,   (3, 3, 3), padding=(1, 1, 1))
        self.bn_quad   = nn.BatchNorm3d(256)
        self.pool      = nn.MaxPool3d(2, 2)

        # After backbone (3 pools of stride-2 on 112px) → 14×14 spatial, T/8 temporal.
        # After fusion pool → 7×7 spatial, T/16 temporal.
        fc_features = 256 * (clip_len // 16) * 7 * 7
        self.dropout   = nn.Dropout(p=0.5)
        self.fc        = nn.Linear(fc_features, num_classes)

        init_vnn_weights(self)

    def forward(self, x):
        lin  = self.bn_lin(self.conv_lin(x))
        quad = self.bn_quad(_quad_unclamped(self.conv_quad(x), self.Q, self.out_ch))
        x = self.pool(lin + quad)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p


# ---------------------------------------------------------------------------
# End-to-end models
# ---------------------------------------------------------------------------

class VNNLegacyFusion(nn.Module):
    """Two-stream legacy VNN matching the original paper architecture.

    Fusion strategy: cat(rgb_features, flow_features) → 192ch.
    No cross-stream product (contrast with VNNFusionHO which adds rgb*flow).

    Args:
        num_classes: Output classes.
        Q: Quadratic rank for the backbone (original: 4). Vary this for Q-sweep.
        Q_fusion: Quadratic rank for the fusion head (original: 2).
        clip_len: Clip frames.
    """

    def __init__(self, num_classes, Q=4, Q_fusion=2, clip_len=16):
        super().__init__()
        self.model_rgb  = LegacyBackbone4Block(num_ch=3, Q=Q)
        self.model_of   = LegacyBackbone4Block(num_ch=2, Q=Q)
        self.model_fuse = LegacyFusionHead(num_classes=num_classes, num_ch=192,
                                           Q=Q_fusion, clip_len=clip_len)

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        return self.model_fuse(torch.cat((out_rgb, out_of), dim=1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.model_fuse.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.model_fuse.fc.parameters():
            if p.requires_grad:
                yield p


class VNNLegacyRgb(nn.Module):
    """RGB-only legacy VNN (single stream, no optical flow).

    Args:
        num_classes: Output classes.
        Q: Quadratic rank for the backbone.
        Q_fusion: Quadratic rank for the classifier head.
        clip_len: Clip frames.
    """

    def __init__(self, num_classes, Q=4, Q_fusion=2, clip_len=16):
        super().__init__()
        self.backbone = LegacyBackbone4Block(num_ch=3, Q=Q)
        self.head     = LegacyFusionHead(num_classes=num_classes, num_ch=96,
                                         Q=Q_fusion, clip_len=clip_len)

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.head.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.head.fc.parameters():
            if p.requires_grad:
                yield p
