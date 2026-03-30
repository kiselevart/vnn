"""
Stable interaction functions for Volterra blocks.

Two drop-in replacements for the raw product in the quadratic/cubic paths.
Both bound the backward gradient analytically — no hard clamps needed.

  'gauss'  — exp(-a²/2)·exp(-b²/2)
             Output in (0,1]. |∂/∂a| ≤ 1/√e ≈ 0.61 everywhere.
             Limitation: always positive, so the quadratic path can only
             add non-negative values to the linear path output.

  'signed' — a·b·exp(-(a²+b²)/4)
             Preserves the sign of the interaction like the raw Volterra
             product. |∂/∂a| = |b(1-a²/2)exp(-(a²+b²)/4)| ≤ 1.04 for
             all (a,b)∈ℝ². Recommended: same geometry as Volterra quadratic,
             analytically bounded gradient, no expressiveness loss.

Models exported:
  lvn_rgb_gauss, lvn_rgb_signed        — single-stream RGB
  lvn_fusion_gauss, lvn_fusion_signed  — two-stream RGB+Flow
"""

import torch
import torch.nn as nn

from .volterra_blocks import ClassifierHead, init_vnn_weights


# ---------------------------------------------------------------------------
# Interaction primitives
# ---------------------------------------------------------------------------

def lvn_gauss(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """exp(-a²/2)·exp(-b²/2). Output in (0,1]. Max |gradient| = 1/√e ≈ 0.61."""
    mid = Q * nch_out
    a = x_conv[:, :mid]
    b = x_conv[:, mid:]
    product = torch.exp(-0.5 * a.pow(2)) * torch.exp(-0.5 * b.pow(2))
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def lvn_signed(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """a·b·exp(-(a²+b²)/4). Signed, bounded gradient ≤ 1.04 everywhere."""
    mid = Q * nch_out
    a = x_conv[:, :mid]
    b = x_conv[:, mid:]
    product = a * b * torch.exp(-0.25 * (a.pow(2) + b.pow(2)))
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def lvn_cubic_signed_sym(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """a²·b·exp(-(a²+b²)/4). Cubic symmetric signed. |∂/∂a| bounded by ≈ 3.3/e."""
    mid = Q * nch_out
    a = x_conv[:, :mid]
    b = x_conv[:, mid:]
    product = a.pow(2) * b * torch.exp(-0.25 * (a.pow(2) + b.pow(2)))
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


def lvn_cubic_signed_gen(x_conv: torch.Tensor, Q: int, nch_out: int) -> torch.Tensor:
    """a·b·c·exp(-(a²+b²+c²)/6). Cubic general signed. |∂/∂a| bounded by ≈ 0.85."""
    chunk = x_conv.shape[1] // 3
    a = x_conv[:, :chunk]
    b = x_conv[:, chunk:2 * chunk]
    c = x_conv[:, 2 * chunk:]
    product = a * b * c * torch.exp(-(a.pow(2) + b.pow(2) + c.pow(2)) / 6.0)
    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1)


# ---------------------------------------------------------------------------
# LaguerreBlock3D
# ---------------------------------------------------------------------------

def soft_clamp(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Smooth surjection onto (-scale, scale). Gradient = sech²(x/scale) > 0 always."""
    return scale * torch.tanh(x / scale)


class LaguerreBlock3D(nn.Module):
    """Volterra block with stable signed/Gaussian interaction instead of raw product.

    Args:
        in_ch, out_ch: Channel dimensions.
        Q:             Quadratic rank.
        Qc:            Cubic rank.
        stride:        MaxPool3d(2) if > 1.
        interaction:   'signed' or 'gauss'.
        cubic_mode:    'symmetric' or 'general' (both use signed-Gaussian cubic).
        use_cubic:     Enable cubic path.
        use_shortcut:  Residual shortcut.
        use_soft_clamp: Apply soft_clamp to block input instead of hard clamp.
        kernel_size:   Conv kernel size.
    """

    _QUAD = {'signed': lvn_signed, 'gauss': lvn_gauss}
    _CUB  = {'symmetric': lvn_cubic_signed_sym, 'general': lvn_cubic_signed_gen}

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4, Qc: int = 2,
                 stride: int = 1, interaction: str = 'signed',
                 cubic_mode: str = 'symmetric', use_cubic: bool = False,
                 use_shortcut: bool = False, use_soft_clamp: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        self.out_ch = out_ch
        self.Q, self.Qc = Q, Qc
        self.use_cubic = use_cubic
        self.use_shortcut = use_shortcut
        self.use_soft_clamp = use_soft_clamp
        self.quad_fn = self._QUAD[interaction]
        self.cub_fn  = self._CUB[cubic_mode]

        ks  = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        pad = tuple(k // 2 for k in ks)

        self.conv_lin  = nn.Conv3d(in_ch, out_ch, ks, padding=pad)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = nn.Conv3d(in_ch, 2 * Q * out_ch, ks, padding=pad)
        self.bn_quad   = nn.BatchNorm3d(out_ch)
        self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        if use_cubic:
            mult = 3 if cubic_mode == 'general' else 2
            self.conv_cubic = nn.Conv3d(in_ch, mult * Qc * out_ch, ks, padding=pad)
            self.bn_cubic   = nn.BatchNorm3d(out_ch)
            self.cubic_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()
        init_vnn_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = soft_clamp(x) if self.use_soft_clamp else x

        out = self.bn_lin(self.conv_lin(x))
        q   = self.bn_quad(self.quad_fn(self.conv_quad(x), self.Q, self.out_ch))
        out = out + self.quad_gate.view(1, -1, 1, 1, 1) * q

        if self.use_cubic:
            c   = self.bn_cubic(self.cub_fn(self.conv_cubic(x), self.Qc, self.out_ch))
            out = out + self.cubic_gate.view(1, -1, 1, 1, 1) * c

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


# ---------------------------------------------------------------------------
# Multi-kernel first block
# ---------------------------------------------------------------------------

class LaguerreMultiKernelBlock3D(nn.Module):
    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, stride: int = 1, interaction: str = 'signed',
                 use_soft_clamp: bool = True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)
        self.use_soft_clamp = use_soft_clamp
        self.quad_fn = LaguerreBlock3D._QUAD[interaction]

        self.lin_convs  = nn.ModuleList()
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.lin_convs.append(nn.Conv3d(in_ch, ch_per_kernel, ks, padding=pad))
            self.quad_convs.append(nn.Conv3d(in_ch, 2 * Q * ch_per_kernel, ks, padding=pad))

        self.bn_lin    = nn.BatchNorm3d(self.out_ch)
        self.bn_quad   = nn.BatchNorm3d(self.out_ch)
        self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-4)
        self.pool      = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()
        init_vnn_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = soft_clamp(x) if self.use_soft_clamp else x
        lin = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))
        quad = self.bn_quad(torch.cat(
            [self.quad_fn(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs],
            dim=1,
        ))
        out = lin + self.quad_gate.view(1, -1, 1, 1, 1) * quad
        return self.pool(out)


# ---------------------------------------------------------------------------
# Backbone, fusion head, end-to-end models
# ---------------------------------------------------------------------------

class LVNBackbone(nn.Module):
    def __init__(self, num_ch: int = 3, interaction: str = 'signed', use_cubic: bool = True):
        super().__init__()
        self.block1 = LaguerreMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5,5,5),(3,3,3),(1,1,1)],
            Q=4, stride=2, interaction=interaction,
        )
        self.block2 = LaguerreBlock3D(24, 32, Q=4, stride=2, interaction=interaction, use_shortcut=True)
        self.block3 = LaguerreBlock3D(32, 64, Q=4, Qc=2, interaction=interaction,
                                      use_cubic=use_cubic, use_shortcut=True)
        self.block4 = LaguerreBlock3D(64, 96, Q=4, Qc=2, stride=2, interaction=interaction,
                                      use_cubic=use_cubic, use_shortcut=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class LVNHead(nn.Module):
    def __init__(self, num_classes: int, num_ch: int = 96,
                 interaction: str = 'signed', use_cubic: bool = True):
        super().__init__()
        self.block = LaguerreBlock3D(num_ch, 256, Q=2, Qc=2, stride=2,
                                     interaction=interaction, use_cubic=use_cubic,
                                     use_shortcut=True)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = ClassifierHead(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.block(x)))

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
    def __init__(self, num_classes: int, interaction: str = 'signed', use_cubic: bool = True,
                 clip_len: int = 16):  # clip_len kept for API compat
        super().__init__()
        self.backbone = LVNBackbone(num_ch=3, interaction=interaction, use_cubic=use_cubic)
        self.head     = LVNHead(num_classes=num_classes, num_ch=96,
                                interaction=interaction, use_cubic=use_cubic)

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
    """Two-stream fusion. Cross term uses signed-Gaussian decay regardless of interaction mode."""

    def __init__(self, num_classes: int, interaction: str = 'signed', use_cubic: bool = True,
                 clip_len: int = 16):
        super().__init__()
        self.model_rgb = LVNBackbone(num_ch=3, interaction=interaction, use_cubic=use_cubic)
        self.model_of  = LVNBackbone(num_ch=2, interaction=interaction, use_cubic=use_cubic)
        self.cross_bn  = nn.BatchNorm3d(96)
        self.head      = LVNHead(num_classes=num_classes, num_ch=288,
                                 interaction=interaction, use_cubic=use_cubic)
        self.cross_abs_max = 0.0

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        # Cross term: signed-Gaussian so gradient is bounded even when streams are large
        decay = torch.exp(-0.25 * (out_rgb.pow(2) + out_of.pow(2)))
        cross = self.cross_bn(out_rgb * out_of * decay)
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


# Named constructors for model_factory
def lvn_rgb_gauss(num_classes: int, clip_len: int = 16)    -> LVNRgb:    return LVNRgb(num_classes, 'gauss',   clip_len=clip_len)
def lvn_rgb_signed(num_classes: int, clip_len: int = 16)   -> LVNRgb:    return LVNRgb(num_classes, 'signed',  clip_len=clip_len)
def lvn_fusion_gauss(num_classes: int, clip_len: int = 16) -> LVNFusion: return LVNFusion(num_classes, 'gauss',  clip_len=clip_len)
def lvn_fusion_signed(num_classes: int, clip_len: int = 16)-> LVNFusion: return LVNFusion(num_classes, 'signed', clip_len=clip_len)
