"""
Proper Laguerre-parameterized convolution for video VNNs.

Core idea
---------
A standard Conv3d learns a free [out_ch, in_ch, T, H, W] kernel.
LaguerreConv3d instead learns coefficients in a *fixed orthogonal basis*:

    kernel[o, i, t, h, w] = Σ_n  coeff[o, i, n, h, w] · φ_n(αt)

where φ_n are the (discretised) generalised Laguerre functions

    φ_n(t) = L_n(t) · exp(-t/2),   t = α · {0, 1, …, T-1}

and L_n is the n-th Laguerre polynomial (computed via the three-term recurrence).

Why this helps
--------------
- Temporal basis is orthogonal → each Laguerre order captures an independent
  frequency band (low-to-high in time).  Gradient paths for different orders
  do not interfere.
- The exponential decay exp(-t/2) builds in a soft causal bias: recent frames
  get more weight than distant ones — appropriate for video.
- Fewer temporal DOF: a T=8 kernel normally has 8 free temporal coefficients
  per (o,i,h,w); with N_lag=4 we compress to 4 while spanning the same space
  up to the N_lag-th moment.

Exports
-------
  LaguerreConv3d               — drop-in Conv3d with Laguerre temporal basis
  LaguerreVolterraBlock3D      — Volterra block using LaguerreConv3d on all paths
  LaguerreBackbone             — 4-block backbone (matches LVNBackbone channel dims)
  LaguerreHead                 — classification head
  LaguerreRgb, LaguFusion     — end-to-end models for ablation
  lvn_laguerre_rgb             — named constructor for model_factory
  lvn_laguerre_fusion          — named constructor for model_factory
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .volterra_blocks import ClassifierHead
from .lvn_blocks import lvn_signed, soft_clamp


# ---------------------------------------------------------------------------
# Laguerre basis construction
# ---------------------------------------------------------------------------

def _laguerre_poly(n: int, t: torch.Tensor) -> torch.Tensor:
    """Evaluate L_n(t) via three-term recurrence.

    L_0 = 1
    L_1 = 1 - t
    L_{n}(t) = ((2n-1-t)·L_{n-1}(t) - (n-1)·L_{n-2}(t)) / n
    """
    if n == 0:
        return torch.ones_like(t)
    L_prev2 = torch.ones_like(t)        # L_0
    L_prev1 = 1.0 - t                   # L_1
    if n == 1:
        return L_prev1
    for k in range(2, n + 1):
        L_curr = ((2 * k - 1 - t) * L_prev1 - (k - 1) * L_prev2) / k
        L_prev2 = L_prev1
        L_prev1 = L_curr
    return L_prev1


def compute_laguerre_basis(T: int, N: int, alpha: float = 1.0) -> torch.Tensor:
    """Compute N normalised Laguerre functions sampled at t = 0, 1, …, T-1.

    φ_n(t) = L_n(αt) · exp(-αt/2)

    Each row is L2-normalised so that ||φ_n||₂ = 1 over the T samples.
    This ensures the Laguerre coefficients have the same scale as the
    original kernel values, making Kaiming init meaningful.

    Returns
    -------
    basis : Tensor [N, T]
    """
    t = torch.arange(T, dtype=torch.float32) * alpha   # [T]
    rows = []
    for n in range(N):
        phi = _laguerre_poly(n, t) * torch.exp(-t / 2.0)   # [T]
        norm = phi.norm()
        rows.append(phi / norm.clamp(min=1e-8))
    return torch.stack(rows, dim=0)   # [N, T]


# ---------------------------------------------------------------------------
# LaguerreConv3d
# ---------------------------------------------------------------------------

class LaguerreConv3d(nn.Module):
    """Conv3d whose temporal kernel dimension is expressed in a Laguerre basis.

    The full spatial+temporal kernel is reconstructed at each forward pass:

        W[o, i, t, h, w] = Σ_n  coeff[o, i, n, h, w] · basis[n, t]

    then used in a standard F.conv3d call.  The basis is fixed (registered as
    a buffer so it moves with .to(device) automatically); only ``coeff`` is
    learned.

    Args:
        in_ch:      Input channels.
        out_ch:     Output channels.
        kernel_size: (T, H, W) kernel size.  Only T (first dim) is Laguerre-
                    parameterised; H and W remain free.
        N_lag:      Number of Laguerre basis functions.  Defaults to T (full
                    expressiveness).  Use N_lag < T to compress temporally.
        alpha:      Laguerre scale.  Larger alpha → faster decay, more
                    localised in time.  Default 1.0 usually works for T≤8.
        padding:    Passed to F.conv3d.
        bias:       If True, adds a learnable bias.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_lag: int | None = None, alpha: float = 1.0,
                 padding=0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size
        if N_lag is None:
            N_lag = T
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.N_lag  = N_lag
        self.padding = padding

        basis = compute_laguerre_basis(T, N_lag, alpha)   # [N_lag, T]
        self.register_buffer("basis", basis)              # fixed

        # Learned: coefficients per (out, in, Laguerre order, H, W)
        self.coeff = nn.Parameter(torch.empty(out_ch, in_ch, N_lag, H, W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None

        self._init_weights()

    def _init_weights(self):
        # Kaiming uniform as if this were a standard conv with fan_in = in_ch*T*H*W
        T, H, W = self.kernel_size
        fan_in = self.in_ch * T * H * W
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(self.coeff, std=std)

    def _get_kernel(self) -> torch.Tensor:
        """Reconstruct the [out_ch, in_ch, T, H, W] kernel from coefficients."""
        # coeff: [O, I, N, H, W]   basis: [N, T]
        # → kernel: [O, I, T, H, W]
        return torch.einsum("oinhw,nt->oithw", self.coeff, self.basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param,
                        padding=self.padding)


# ---------------------------------------------------------------------------
# LaguerreVolterraBlock3D
# ---------------------------------------------------------------------------

class LaguerreVolterraBlock3D(nn.Module):
    """Volterra block where every convolution uses a Laguerre temporal basis.

    Linear path   : LaguerreConv3d(in_ch, out_ch)
    Quadratic path: LaguerreConv3d(in_ch, 2*Q*out_ch) → lvn_signed interaction

    Args:
        in_ch, out_ch : Channel dimensions.
        Q             : Quadratic rank.
        kernel_size   : (T, H, W) or int.
        N_lag         : Laguerre basis size (temporal DOF).  None = T (full).
        alpha         : Laguerre time scale.
        stride        : MaxPool3d(2) if > 1.
        use_shortcut  : Residual 1×1 shortcut.
        use_soft_clamp: Smooth input clamp (tanh-based).
    """

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4,
                 kernel_size=3, N_lag: int | None = None, alpha: float = 1.0,
                 stride: int = 1, use_shortcut: bool = False,
                 use_soft_clamp: bool = True):
        super().__init__()
        self.out_ch  = out_ch
        self.Q       = Q
        self.use_shortcut   = use_shortcut
        self.use_soft_clamp = use_soft_clamp

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        pad = tuple(k // 2 for k in kernel_size)

        self.conv_lin  = LaguerreConv3d(in_ch, out_ch, kernel_size, N_lag, alpha, padding=pad)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = LaguerreConv3d(in_ch, 2 * Q * out_ch, kernel_size, N_lag, alpha, padding=pad)
        self.bn_quad   = nn.BatchNorm3d(out_ch)
        self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()
        # BN init (coeff already Kaiming-inited in LaguerreConv3d)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = soft_clamp(x) if self.use_soft_clamp else x
        out = self.bn_lin(self.conv_lin(x))
        q   = self.bn_quad(lvn_signed(self.conv_quad(x), self.Q, self.out_ch))
        out = out + self.quad_gate.view(1, -1, 1, 1, 1) * q
        if self.use_shortcut:
            out = self.shortcut(x) + out
        return self.pool(out)


# ---------------------------------------------------------------------------
# Multi-kernel first block (same topology as LaguerreMultiKernelBlock3D)
# ---------------------------------------------------------------------------

class LaguerreMultiKernelBlock3D(nn.Module):
    """Multi-scale first block using LaguerreConv3d for each kernel branch."""

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, N_lag: int | None = None, alpha: float = 1.0,
                 stride: int = 1, use_soft_clamp: bool = True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)
        self.use_soft_clamp = use_soft_clamp

        self.lin_convs  = nn.ModuleList()
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.lin_convs.append(
                LaguerreConv3d(in_ch, ch_per_kernel, ks, N_lag, alpha, padding=pad))
            self.quad_convs.append(
                LaguerreConv3d(in_ch, 2 * Q * ch_per_kernel, ks, N_lag, alpha, padding=pad))

        self.bn_lin    = nn.BatchNorm3d(self.out_ch)
        self.bn_quad   = nn.BatchNorm3d(self.out_ch)
        self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-4)
        self.pool      = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = soft_clamp(x) if self.use_soft_clamp else x
        lin  = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))
        quad = self.bn_quad(torch.cat(
            [lvn_signed(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs],
            dim=1,
        ))
        out  = lin + self.quad_gate.view(1, -1, 1, 1, 1) * quad
        return self.pool(out)


# ---------------------------------------------------------------------------
# Backbone, head, end-to-end models
# ---------------------------------------------------------------------------

class LaguerreBackbone(nn.Module):
    """4-block backbone matching LVNBackbone channel layout.

    Output: [B, 96, T/8, H/8, W/8]  (same as LVNBackbone)
    """

    def __init__(self, num_ch: int = 3, N_lag: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.block1 = LaguerreMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, N_lag=N_lag, alpha=alpha, stride=2,
        )
        self.block2 = LaguerreVolterraBlock3D(
            24, 32, Q=4, kernel_size=3, N_lag=N_lag, alpha=alpha,
            stride=2, use_shortcut=True,
        )
        self.block3 = LaguerreVolterraBlock3D(
            32, 64, Q=4, kernel_size=3, N_lag=N_lag, alpha=alpha,
            use_shortcut=True,
        )
        self.block4 = LaguerreVolterraBlock3D(
            64, 96, Q=4, kernel_size=3, N_lag=N_lag, alpha=alpha,
            stride=2, use_shortcut=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class LaguerreHead(nn.Module):
    def __init__(self, num_classes: int, num_ch: int = 96,
                 N_lag: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.block = LaguerreVolterraBlock3D(
            num_ch, 256, Q=2, kernel_size=3, N_lag=N_lag, alpha=alpha,
            stride=2, use_shortcut=True,
        )
        self.classifier = ClassifierHead(12544, num_classes)

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


class LaguerreRgb(nn.Module):
    def __init__(self, num_classes: int, N_lag: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.backbone = LaguerreBackbone(num_ch=3, N_lag=N_lag, alpha=alpha)
        self.head     = LaguerreHead(num_classes=num_classes, N_lag=N_lag, alpha=alpha)

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


class LaguFusion(nn.Module):
    """Two-stream Laguerre-VNN fusion.  Cross term uses signed-Gaussian decay."""

    def __init__(self, num_classes: int, N_lag: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.model_rgb = LaguerreBackbone(num_ch=3, N_lag=N_lag, alpha=alpha)
        self.model_of  = LaguerreBackbone(num_ch=2, N_lag=N_lag, alpha=alpha)
        self.cross_bn  = nn.BatchNorm3d(96)
        self.head      = LaguerreHead(num_classes=num_classes, num_ch=288,
                                      N_lag=N_lag, alpha=alpha)
        self.cross_abs_max = 0.0

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        decay   = torch.exp(-0.25 * (out_rgb.pow(2) + out_of.pow(2)))
        cross   = self.cross_bn(out_rgb * out_of * decay)
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
def lvn_laguerre_rgb(num_classes: int) -> LaguerreRgb:
    return LaguerreRgb(num_classes)

def lvn_laguerre_fusion(num_classes: int) -> LaguFusion:
    return LaguFusion(num_classes)
