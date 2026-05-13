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

from .volterra_blocks import ClassifierHead, volterra_quadratic


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
    """Volterra block with either a Laguerre temporal basis or a free (monomial) Conv3d.

    Setting use_laguerre_basis=False gives a standard Conv3d with the same
    signed interaction — the only variable between the two is the temporal
    kernel parameterisation, making it a clean ablation target.

    Linear path   : LaguerreConv3d or Conv3d (in_ch, out_ch)
    Quadratic path: same conv type (in_ch, 2*Q*out_ch) → volterra_quadratic interaction

    Args:
        in_ch, out_ch      : Channel dimensions.
        Q                  : Quadratic rank.
        kernel_size        : (T, H, W) or int.
        N_lag              : Laguerre basis size. None = T (full). Ignored when
                             use_laguerre_basis=False.
        alpha              : Laguerre time scale.
        stride             : MaxPool3d(2) if > 1.
        use_shortcut       : Residual 1×1 shortcut.
        use_soft_clamp     : Smooth input clamp (tanh-based).
        use_laguerre_basis : If False, uses plain Conv3d (monomial baseline).
    """

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4,
                 kernel_size=3, N_lag: int | None = None, alpha: float = 1.0,
                 stride: int = 1, use_shortcut: bool = False,
                 use_laguerre_basis: bool = True):
        super().__init__()
        self.out_ch  = out_ch
        self.Q       = Q
        self.use_shortcut   = use_shortcut

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        pad = tuple(k // 2 for k in kernel_size)

        def _make_conv(out_channels):
            if use_laguerre_basis:
                return LaguerreConv3d(in_ch, out_channels, kernel_size, N_lag, alpha, padding=pad)
            else:
                return nn.Conv3d(in_ch, out_channels, kernel_size, padding=pad)

        self.conv_lin  = _make_conv(out_ch)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = _make_conv(2 * Q * out_ch)
        self.bn_quad   = nn.BatchNorm3d(out_ch)
        self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-2)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()
        if not use_laguerre_basis:
            # LaguerreConv3d self-inits; plain Conv3d needs explicit Kaiming
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clamp(-50.0, 50.0)
        out = self.bn_lin(self.conv_lin(x))
        q   = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
        out = out + self.quad_gate.view(1, -1, 1, 1, 1) * q
        if self.use_shortcut:
            out = self.shortcut(x) + out
        return self.pool(out)


# ---------------------------------------------------------------------------
# Multi-kernel first block (same topology as LaguerreMultiKernelBlock3D)
# ---------------------------------------------------------------------------

class LaguerreMultiKernelBlock3D(nn.Module):
    """Multi-scale first block; use_laguerre_basis controls Conv3d vs LaguerreConv3d."""

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, N_lag: int | None = None, alpha: float = 1.0,
                 stride: int = 1, use_laguerre_basis: bool = True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)

        self.lin_convs  = nn.ModuleList()
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            if use_laguerre_basis:
                self.lin_convs.append(
                    LaguerreConv3d(in_ch, ch_per_kernel, ks, N_lag, alpha, padding=pad))
                self.quad_convs.append(
                    LaguerreConv3d(in_ch, 2 * Q * ch_per_kernel, ks, N_lag, alpha, padding=pad))
            else:
                self.lin_convs.append(nn.Conv3d(in_ch, ch_per_kernel, ks, padding=pad))
                self.quad_convs.append(nn.Conv3d(in_ch, 2 * Q * ch_per_kernel, ks, padding=pad))

        self.bn_lin    = nn.BatchNorm3d(self.out_ch)
        self.bn_quad   = nn.BatchNorm3d(self.out_ch)
        self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-2)
        self.pool      = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        if not use_laguerre_basis:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = x.clamp(-50.0, 50.0)
        lin  = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))
        quad = self.bn_quad(torch.cat(
            [volterra_quadratic(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs],
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

    def __init__(self, num_ch: int = 3, N_lag: int | None = None, alpha: float = 1.0,
                 use_laguerre_basis: bool = True):
        super().__init__()
        kw = dict(N_lag=N_lag, alpha=alpha, use_laguerre_basis=use_laguerre_basis)
        self.block1 = LaguerreMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2, **kw,
        )
        self.block2 = LaguerreVolterraBlock3D(24, 32, Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)
        self.block3 = LaguerreVolterraBlock3D(32, 64, Q=4, kernel_size=3, use_shortcut=True, **kw)
        self.block4 = LaguerreVolterraBlock3D(64, 96, Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class LaguerreHead(nn.Module):
    def __init__(self, num_classes: int, num_ch: int = 96,
                 N_lag: int | None = None, alpha: float = 1.0,
                 clip_len: int = 16, use_laguerre_basis: bool = True):
        super().__init__()
        self.block = LaguerreVolterraBlock3D(
            num_ch, 256, Q=2, kernel_size=3, N_lag=N_lag, alpha=alpha,
            stride=2, use_shortcut=True, use_laguerre_basis=use_laguerre_basis,
        )
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


class LaguerreRgb(nn.Module):
    def __init__(self, num_classes: int, N_lag: int | None = None, alpha: float = 1.0,
                 clip_len: int = 16, use_laguerre_basis: bool = True):
        super().__init__()
        kw = dict(N_lag=N_lag, alpha=alpha, use_laguerre_basis=use_laguerre_basis)
        self.backbone = LaguerreBackbone(num_ch=3, **kw)
        self.head     = LaguerreHead(num_classes=num_classes, clip_len=clip_len, **kw)

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
    """Two-stream Laguerre-VNN fusion.  Cross term is a pure Volterra quadratic product."""

    def __init__(self, num_classes: int, N_lag: int | None = None, alpha: float = 1.0,
                 clip_len: int = 16, use_laguerre_basis: bool = True):
        super().__init__()
        kw = dict(N_lag=N_lag, alpha=alpha, use_laguerre_basis=use_laguerre_basis)
        self.model_rgb = LaguerreBackbone(num_ch=3, **kw)
        self.model_of  = LaguerreBackbone(num_ch=2, **kw)
        self.cross_bn  = nn.BatchNorm3d(96)
        self.head      = LaguerreHead(num_classes=num_classes, num_ch=288,
                                      clip_len=clip_len, **kw)
        self.cross_abs_max = 0.0

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        cross   = self.cross_bn((out_rgb * out_of).clamp(-50.0, 50.0))
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
def lvn_laguerre_rgb(num_classes: int, clip_len: int = 16,
                     n_lag: int | None = None) -> LaguerreRgb:
    return LaguerreRgb(num_classes, N_lag=n_lag, clip_len=clip_len, use_laguerre_basis=True)

def lvn_laguerre_fusion(num_classes: int, clip_len: int = 16,
                        n_lag: int | None = None) -> LaguFusion:
    return LaguFusion(num_classes, N_lag=n_lag, clip_len=clip_len, use_laguerre_basis=True)

def lvn_monomial_rgb(num_classes: int, clip_len: int = 16) -> LaguerreRgb:
    """Identical to lvn_laguerre_rgb but uses free Conv3d instead of LaguerreConv3d.
    Direct ablation: only the temporal basis differs."""
    return LaguerreRgb(num_classes, clip_len=clip_len, use_laguerre_basis=False)

def lvn_monomial_fusion(num_classes: int, clip_len: int = 16) -> LaguFusion:
    return LaguFusion(num_classes, clip_len=clip_len, use_laguerre_basis=False)


# ---------------------------------------------------------------------------
# Full 3D Laguerre basis — Tucker decomposition over T, H, W
# ---------------------------------------------------------------------------

def _compute_laguerre_basis_spatial(size: int, N: int, alpha: float = 1.0,
                                     center: bool = True) -> torch.Tensor:
    """Laguerre basis for a spatial dimension.

    center=True  — evaluate at |pos - size//2| * alpha, giving symmetric
                   (even) responses.  Effective rank ≤ (size+1)//2, so N
                   should not exceed that (e.g. N≤2 for size=3).
    center=False — positions {0,...,size-1} * alpha, same as temporal.
    """
    pos = torch.arange(size, dtype=torch.float32)
    t = (pos - size // 2).abs() * alpha if center else pos * alpha
    rows = []
    for n in range(N):
        phi = _laguerre_poly(n, t) * torch.exp(-t / 2.0)
        norm = phi.norm()
        rows.append(phi / norm.clamp(min=1e-8))
    return torch.stack(rows, dim=0)   # [N, size]


class LaguerreConv3d_Full(nn.Module):
    """Conv3d with independent Laguerre bases along all three (T, H, W) dimensions.

    Kernel reconstructed via Tucker decomposition with fixed orthogonal bases:

        W[o,i,t,h,w] = Σ_{n,m,k} coeff[o,i,n,m,k] · φ_T[n,t] · φ_H[m,h] · φ_W[k,w]

    Parameter count per (out, in) pair: N_T × N_H × N_W  vs  T × H × W for Conv3d.

    Args:
        in_ch, out_ch:  Channel dimensions.
        kernel_size:    (T, H, W) tuple or int.
        N_lag_T:        Laguerre orders for temporal dim. Default = T (no compression).
        N_lag_H:        Laguerre orders for H. Default = H.
        N_lag_W:        Laguerre orders for W. Default = W.
        alpha_T:        Laguerre scale for T (causal decay). Default 1.0.
        alpha_H:        Laguerre scale for H. Smaller → more uniform spatial coverage.
        alpha_W:        Laguerre scale for W.
        center_spatial: Evaluate H/W bases at distance from kernel centre for
                        symmetric responses. Rank limited to (size+1)//2 per dim;
                        set N_lag_H/W accordingly (≤2 for 3×3).  Default True.
        padding, bias:  Passed to F.conv3d.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_lag_T: int | None = None, N_lag_H: int | None = None,
                 N_lag_W: int | None = None,
                 alpha_T: float = 1.0, alpha_H: float = 1.0, alpha_W: float = 1.0,
                 center_spatial: bool = True, padding=0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size
        N_T = N_lag_T if N_lag_T is not None else T
        N_H = N_lag_H if N_lag_H is not None else H
        N_W = N_lag_W if N_lag_W is not None else W

        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.register_buffer("basis_T", compute_laguerre_basis(T, N_T, alpha_T))
        self.register_buffer("basis_H", _compute_laguerre_basis_spatial(H, N_H, alpha_H, center_spatial))
        self.register_buffer("basis_W", _compute_laguerre_basis_spatial(W, N_W, alpha_W, center_spatial))

        self.coeff = nn.Parameter(torch.empty(out_ch, in_ch, N_T, N_H, N_W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self._init_weights()

    def _init_weights(self):
        T, H, W = self.kernel_size
        std = math.sqrt(2.0 / (self.in_ch * T * H * W))
        nn.init.normal_(self.coeff, std=std)

    def _get_kernel(self) -> torch.Tensor:
        """Tucker reconstruction of [O, I, T, H, W] kernel — sequential contraction."""
        k = torch.einsum("oinmk,nt->oitmk", self.coeff,  self.basis_T)
        k = torch.einsum("oitmk,mh->oithk", k,           self.basis_H)
        k = torch.einsum("oithk,kw->oithw", k,           self.basis_W)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param, padding=self.padding)


class LaguerreFullVolterraBlock3D(nn.Module):
    """Volterra block using LaguerreConv3d_Full on all conv paths.

    Drop-in replacement for LaguerreVolterraBlock3D; only conv type differs.

    Args:
        in_ch, out_ch:   Channel dimensions.
        Q:               Quadratic rank.
        kernel_size:     (T, H, W) or int.
        N_lag_T:         Laguerre orders for T. None = T (no compression).
        N_lag_S:         Laguerre orders for H and W (shared). None = no compression.
        alpha_T:         Temporal Laguerre scale.
        alpha_S:         Spatial Laguerre scale.
        center_spatial:  Symmetric spatial basis. Default True.
        stride:          MaxPool3d(2) if > 1.
        use_shortcut:    Residual 1×1 shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4,
                 kernel_size=3, N_lag_T: int | None = None, N_lag_S: int | None = None,
                 alpha_T: float = 1.0, alpha_S: float = 1.0,
                 center_spatial: bool = True,
                 stride: int = 1, use_shortcut: bool = False):
        super().__init__()
        self.out_ch = out_ch
        self.Q = Q
        self.use_shortcut = use_shortcut

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        pad = tuple(k // 2 for k in kernel_size)

        def _make_conv(out_channels):
            return LaguerreConv3d_Full(
                in_ch, out_channels, kernel_size,
                N_lag_T=N_lag_T, N_lag_H=N_lag_S, N_lag_W=N_lag_S,
                alpha_T=alpha_T, alpha_H=alpha_S, alpha_W=alpha_S,
                center_spatial=center_spatial, padding=pad,
            )

        self.conv_lin  = _make_conv(out_ch)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = _make_conv(2 * Q * out_ch)
        self.bn_quad   = nn.BatchNorm3d(out_ch)
        self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-2)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clamp(-50.0, 50.0)
        out = self.bn_lin(self.conv_lin(x))
        q   = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
        out = out + self.quad_gate.view(1, -1, 1, 1, 1) * q
        if self.use_shortcut:
            out = self.shortcut(x) + out
        return self.pool(out)


class LaguerreFullMultiKernelBlock3D(nn.Module):
    """Multi-scale first block using LaguerreConv3d_Full on all conv paths."""

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, N_lag_T: int | None = None, N_lag_S: int | None = None,
                 alpha_T: float = 1.0, alpha_S: float = 1.0,
                 center_spatial: bool = True, stride: int = 1):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)

        self.lin_convs  = nn.ModuleList()
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            kw = dict(N_lag_T=N_lag_T, N_lag_H=N_lag_S, N_lag_W=N_lag_S,
                      alpha_T=alpha_T, alpha_H=alpha_S, alpha_W=alpha_S,
                      center_spatial=center_spatial, padding=pad)
            self.lin_convs.append(LaguerreConv3d_Full(in_ch, ch_per_kernel, ks, **kw))
            self.quad_convs.append(LaguerreConv3d_Full(in_ch, 2 * Q * ch_per_kernel, ks, **kw))

        self.bn_lin    = nn.BatchNorm3d(self.out_ch)
        self.bn_quad   = nn.BatchNorm3d(self.out_ch)
        self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-2)
        self.pool      = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = x.clamp(-50.0, 50.0)
        lin  = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))
        quad = self.bn_quad(torch.cat(
            [volterra_quadratic(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs],
            dim=1,
        ))
        return self.pool(lin + self.quad_gate.view(1, -1, 1, 1, 1) * quad)


class LaguerreFullBackbone(nn.Module):
    """4-block backbone using LaguerreConv3d_Full (Laguerre basis on T, H, W).

    Output: [B, 96, T/8, H/8, W/8]  (same as LaguerreBackbone)

    Args:
        num_ch:         Input channels.
        N_lag_T:        Temporal Laguerre orders. None = no compression.
        N_lag_S:        Spatial Laguerre orders (H and W). None = no compression.
                        Typical: 2 for 3×3 kernels (center_spatial=True gives rank ≤2).
        alpha_T:        Temporal Laguerre scale. Default 1.0.
        alpha_S:        Spatial Laguerre scale. Smaller → more uniform coverage.
        center_spatial: Symmetric spatial basis. Default True.
    """

    def __init__(self, num_ch: int = 3, N_lag_T: int | None = None,
                 N_lag_S: int | None = None, alpha_T: float = 1.0,
                 alpha_S: float = 1.0, center_spatial: bool = True):
        super().__init__()
        kw = dict(N_lag_T=N_lag_T, N_lag_S=N_lag_S, alpha_T=alpha_T,
                  alpha_S=alpha_S, center_spatial=center_spatial)
        self.block1 = LaguerreFullMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2, **kw,
        )
        self.block2 = LaguerreFullVolterraBlock3D(24, 32,  Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)
        self.block3 = LaguerreFullVolterraBlock3D(32, 64,  Q=4, kernel_size=3,           use_shortcut=True, **kw)
        self.block4 = LaguerreFullVolterraBlock3D(64, 96,  Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class LaguerreFullHead(nn.Module):
    def __init__(self, num_classes: int, num_ch: int = 96,
                 N_lag_T: int | None = None, N_lag_S: int | None = None,
                 alpha_T: float = 1.0, alpha_S: float = 1.0,
                 center_spatial: bool = True, clip_len: int = 16):
        super().__init__()
        self.block = LaguerreFullVolterraBlock3D(
            num_ch, 256, Q=2, kernel_size=3, stride=2, use_shortcut=True,
            N_lag_T=N_lag_T, N_lag_S=N_lag_S, alpha_T=alpha_T, alpha_S=alpha_S,
            center_spatial=center_spatial,
        )
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


class LaguerreFullRgb(nn.Module):
    def __init__(self, num_classes: int, N_lag_T: int | None = None,
                 N_lag_S: int | None = None, alpha_T: float = 1.0,
                 alpha_S: float = 1.0, center_spatial: bool = True,
                 clip_len: int = 16):
        super().__init__()
        kw = dict(N_lag_T=N_lag_T, N_lag_S=N_lag_S, alpha_T=alpha_T,
                  alpha_S=alpha_S, center_spatial=center_spatial)
        self.backbone = LaguerreFullBackbone(num_ch=3, **kw)
        self.head     = LaguerreFullHead(num_classes=num_classes, clip_len=clip_len, **kw)

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


class LaguerreFullFusion(nn.Module):
    """Two-stream fusion with full 3D Laguerre basis on both streams."""

    def __init__(self, num_classes: int, N_lag_T: int | None = None,
                 N_lag_S: int | None = None, alpha_T: float = 1.0,
                 alpha_S: float = 1.0, center_spatial: bool = True,
                 clip_len: int = 16):
        super().__init__()
        kw = dict(N_lag_T=N_lag_T, N_lag_S=N_lag_S, alpha_T=alpha_T,
                  alpha_S=alpha_S, center_spatial=center_spatial)
        self.model_rgb = LaguerreFullBackbone(num_ch=3, **kw)
        self.model_of  = LaguerreFullBackbone(num_ch=2, **kw)
        self.cross_bn  = nn.BatchNorm3d(96)
        self.head      = LaguerreFullHead(num_classes=num_classes, num_ch=288,
                                          clip_len=clip_len, **kw)
        self.cross_abs_max = 0.0

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        cross   = self.cross_bn((out_rgb * out_of).clamp(-50.0, 50.0))
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


def lvn_laguerre_full_rgb(num_classes: int, clip_len: int = 16,
                           n_lag_t: int | None = None,
                           n_lag_s: int | None = None) -> LaguerreFullRgb:
    return LaguerreFullRgb(num_classes, N_lag_T=n_lag_t, N_lag_S=n_lag_s, clip_len=clip_len)

def lvn_laguerre_full_fusion(num_classes: int, clip_len: int = 16,
                              n_lag_t: int | None = None,
                              n_lag_s: int | None = None) -> LaguerreFullFusion:
    return LaguerreFullFusion(num_classes, N_lag_T=n_lag_t, N_lag_S=n_lag_s, clip_len=clip_len)
