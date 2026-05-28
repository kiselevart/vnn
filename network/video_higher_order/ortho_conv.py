"""
Orthogonal polynomial basis convolutions for video VNNs.

Extends the Laguerre approach (laguerre_conv.py) to other polynomial families.
Each family parameterises the temporal dimension of a Conv3d kernel as a linear
combination of fixed orthogonal basis functions; the spatial (H, W) dims remain
free.  Only the ``coeff`` tensor is learned.

Supported families
------------------
  'laguerre'  : Laguerre — causal/asymmetric, exponential decay toward past frames.
                Delegates to compute_laguerre_basis from laguerre_conv.
  'legendre'  : Legendre — uniform temporal weighting; equal attention to all frames.
  'chebyshev' : Chebyshev T₁ — same domain as Legendre but Chebyshev node emphasis;
                better uniform approximation for the same N.
  'hermite'   : Hermite (physicists') — Gaussian envelope centred at mid-clip;
                symmetric past/future decay, centre-biased.

Architecture
------------
  OrthoConv3d              — basis-parameterised Conv3d (temporal dim only)
  OrthoVolterraBlock3D     — Volterra block (linear + quadratic) using OrthoConv3d
  OrthoMultiKernelBlock3D  — multi-scale first block
  OrthoBackbone            — 4-block backbone, output [B, 96, T/8, H/8, W/8]
  OrthoHead                — classification head
  OrthoRgb                 — single-stream RGB model
  OrthoFusion              — two-stream (RGB + optical flow) fusion model

Named constructors for model_factory
--------------------------------------
  lvn_legendre_rgb  / lvn_legendre_fusion
  lvn_chebyshev_rgb / lvn_chebyshev_fusion
  lvn_hermite_rgb   / lvn_hermite_fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .volterra_blocks import ClassifierHead, volterra_quadratic
from .laguerre_conv import compute_laguerre_basis


# ---------------------------------------------------------------------------
# Basis computation
# ---------------------------------------------------------------------------

def _compute_legendre_basis(T: int, N: int) -> torch.Tensor:
    """Legendre polynomial basis [N, T], L2-normalised.

    Polynomials P_n evaluated at T evenly-spaced points mapped to [-1, 1]:
        x_t = 2t / (T−1) − 1    (when T=1, x_t = 0 for all t)

    Recurrence:
        P_0(x) = 1
        P_1(x) = x
        P_n(x) = ((2n−1)·x·P_{n−1} − (n−1)·P_{n−2}) / n
    """
    t = torch.arange(T, dtype=torch.float64)
    x = 2.0 * t / max(T - 1, 1) - 1.0   # [T] in [-1, 1]

    rows: list[torch.Tensor] = []
    P_prev2 = torch.ones(T, dtype=torch.float64)   # P_0
    P_prev1 = x.clone()                             # P_1

    for n in range(N):
        if n == 0:
            poly = P_prev2
        elif n == 1:
            poly = P_prev1
        else:
            poly = ((2 * n - 1) * x * P_prev1 - (n - 1) * P_prev2) / n
            P_prev2 = P_prev1
            P_prev1 = poly
        norm = poly.norm()
        rows.append((poly / norm.clamp(min=1e-8)).float())

    return torch.stack(rows, dim=0)   # [N, T]


def _compute_chebyshev_basis(T: int, N: int) -> torch.Tensor:
    """Chebyshev Type-1 polynomial basis [N, T], L2-normalised.

    T_n(x) = cos(n · arccos(x)),  x ∈ [-1, 1].
    Closed-form trig avoids recurrence instability for large n.
    Domain mapping: same as Legendre (x = 2t/(T−1) − 1).
    """
    t = torch.arange(T, dtype=torch.float64)
    x = 2.0 * t / max(T - 1, 1) - 1.0
    # Clamp slightly inside (-1, 1) to avoid arccos(±1) edge-case on grad
    theta = torch.acos(x.clamp(-1.0 + 1e-7, 1.0 - 1e-7))   # [T]

    rows: list[torch.Tensor] = []
    for n in range(N):
        poly = torch.cos(n * theta)
        norm = poly.norm()
        rows.append((poly / norm.clamp(min=1e-8)).float())

    return torch.stack(rows, dim=0)   # [N, T]


def _compute_hermite_basis(T: int, N: int, alpha: float = 1.0) -> torch.Tensor:
    """Hermite (physicists') polynomial basis [N, T], L2-normalised.

    Basis functions: φ_n(t) = H_n(α·(t − centre)) · exp(−α²·(t−centre)²/2)

    where centre = (T−1)/2, so the Gaussian envelope is symmetric around the
    mid-frame.  The recurrence is:
        H_0(s) = 1
        H_1(s) = 2s
        H_n(s) = 2s·H_{n−1}(s) − 2(n−1)·H_{n−2}(s)

    alpha controls bandwidth: larger → narrower Gaussian → more temporal
    localisation around the centre frame.  Default 1.0 matches the Laguerre
    convention.
    """
    t = torch.arange(T, dtype=torch.float64)
    s = alpha * (t - (T - 1) / 2.0)   # centred argument [T]
    env = torch.exp(-s ** 2 / 2.0)    # Gaussian envelope

    rows: list[torch.Tensor] = []
    H_prev2 = torch.ones(T, dtype=torch.float64)   # H_0
    H_prev1 = 2.0 * s                               # H_1

    for n in range(N):
        if n == 0:
            Hn = H_prev2
        elif n == 1:
            Hn = H_prev1
        else:
            Hn = 2.0 * s * H_prev1 - 2.0 * (n - 1) * H_prev2
            H_prev2 = H_prev1
            H_prev1 = Hn
        phi = Hn * env
        norm = phi.norm()
        rows.append((phi / norm.clamp(min=1e-8)).float())

    return torch.stack(rows, dim=0)   # [N, T]


def compute_ortho_basis(T: int, N: int, kind: str,
                        alpha: float = 1.0) -> torch.Tensor:
    """Compute N L2-normalised orthogonal basis functions sampled at T points.

    Args:
        T:     Number of temporal points (temporal kernel extent).
        N:     Number of basis functions.
        kind:  Polynomial family — 'laguerre', 'legendre', 'chebyshev', 'hermite'.
        alpha: Scale parameter (family-dependent):
               - laguerre : temporal decay rate (larger → faster decay).
               - hermite  : Gaussian bandwidth (larger → narrower envelope).
               - legendre / chebyshev : ignored (domain is always [-1, 1]).

    Returns:
        basis : Tensor [N, T], float32.
    """
    if kind == 'laguerre':
        return compute_laguerre_basis(T, N, alpha)
    elif kind == 'legendre':
        return _compute_legendre_basis(T, N)
    elif kind == 'chebyshev':
        return _compute_chebyshev_basis(T, N)
    elif kind == 'hermite':
        return _compute_hermite_basis(T, N, alpha)
    else:
        raise ValueError(
            f"Unknown basis kind {kind!r}. "
            "Choose from: 'laguerre', 'legendre', 'chebyshev', 'hermite'."
        )


# ---------------------------------------------------------------------------
# OrthoConv3d
# ---------------------------------------------------------------------------

class OrthoConv3d(nn.Module):
    """Conv3d whose temporal kernel dimension is expressed in an orthogonal basis.

    Learned weights are coefficients [out_ch, in_ch, N_poly, H, W].
    Full kernel is reconstructed at each forward pass:

        W[o, i, t, h, w] = Σ_n  coeff[o, i, n, h, w] · basis[n, t]

    The basis is a fixed registered buffer; only ``coeff`` is learned.
    The name ``coeff`` is intentional: the train_par.py optimizer hook
    (``is_laguerre_coeff``) detects ``.coeff``-named parameters and applies
    the higher LR multiplier automatically.

    Args:
        in_ch, out_ch : Channel dimensions.
        kernel_size   : (T, H, W) or int. Only T is basis-parameterised.
        N_poly        : Number of basis functions. None → T (no compression).
        alpha         : Scale for Laguerre/Hermite; ignored for Legendre/Chebyshev.
        basis_kind    : 'laguerre', 'legendre', 'chebyshev', or 'hermite'.
        padding, bias : Passed to F.conv3d.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_poly: int | None = None, alpha: float = 1.0,
                 basis_kind: str = 'laguerre',
                 padding: int | tuple[int, ...] = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size
        if N_poly is None:
            N_poly = T

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.N_poly = N_poly
        self.padding = padding

        basis = compute_ortho_basis(T, N_poly, basis_kind, alpha)
        self.register_buffer("basis", basis)          # [N_poly, T], fixed

        self.coeff = nn.Parameter(torch.empty(out_ch, in_ch, N_poly, H, W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self._init_weights()

    def _init_weights(self):
        T, H, W = self.kernel_size
        std = math.sqrt(2.0 / (self.in_ch * T * H * W))
        nn.init.normal_(self.coeff, std=std)

    def _get_kernel(self) -> torch.Tensor:
        # coeff: [O, I, N, H, W]   basis: [N, T]  →  kernel: [O, I, T, H, W]
        return torch.einsum("oinhw,nt->oithw", self.coeff, self.basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param,
                        padding=self.padding)


# ---------------------------------------------------------------------------
# OrthoVolterraBlock3D
# ---------------------------------------------------------------------------

class OrthoVolterraBlock3D(nn.Module):
    """Volterra block (linear + quadratic) using OrthoConv3d on all paths.

    Interface mirrors LaguerreVolterraBlock3D.

    Args:
        in_ch, out_ch : Channel dimensions.
        Q             : Quadratic CP rank.
        kernel_size   : (T, H, W) or int.
        N_poly        : Basis functions. None → T (full expressiveness).
        alpha         : Scale (Laguerre/Hermite); ignored for Legendre/Chebyshev.
        basis_kind    : Polynomial family.
        stride        : MaxPool3d(2) applied when > 1.
        use_shortcut  : Residual 1×1×1 shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4,
                 kernel_size=3, N_poly: int | None = None, alpha: float = 1.0,
                 basis_kind: str = 'laguerre',
                 stride: int = 1, use_shortcut: bool = False):
        super().__init__()
        self.out_ch = out_ch
        self.Q = Q
        self.use_shortcut = use_shortcut

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        pad = tuple(k // 2 for k in kernel_size)

        def _make(out_channels: int) -> OrthoConv3d:
            return OrthoConv3d(in_ch, out_channels, kernel_size,
                               N_poly=N_poly, alpha=alpha,
                               basis_kind=basis_kind, padding=pad)

        self.conv_lin  = _make(out_ch)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = _make(2 * Q * out_ch)
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


# ---------------------------------------------------------------------------
# OrthoMultiKernelBlock3D
# ---------------------------------------------------------------------------

class OrthoMultiKernelBlock3D(nn.Module):
    """Multi-scale first block using OrthoConv3d.

    Parallel kernels at different sizes; outputs concatenated then BN+gate.
    """

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, N_poly: int | None = None, alpha: float = 1.0,
                 basis_kind: str = 'laguerre', stride: int = 1):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)

        self.lin_convs: nn.ModuleList = nn.ModuleList()
        self.quad_convs: nn.ModuleList = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.lin_convs.append(
                OrthoConv3d(in_ch, ch_per_kernel, ks,
                            N_poly=N_poly, alpha=alpha,
                            basis_kind=basis_kind, padding=pad))
            self.quad_convs.append(
                OrthoConv3d(in_ch, 2 * Q * ch_per_kernel, ks,
                            N_poly=N_poly, alpha=alpha,
                            basis_kind=basis_kind, padding=pad))

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
            [volterra_quadratic(c(x), self.Q, self.ch_per_kernel)
             for c in self.quad_convs],
            dim=1,
        ))
        return self.pool(lin + self.quad_gate.view(1, -1, 1, 1, 1) * quad)


# ---------------------------------------------------------------------------
# Backbone / Head / Models
# ---------------------------------------------------------------------------

class OrthoBackbone(nn.Module):
    """4-block backbone parameterised by an orthogonal polynomial basis.

    Channel layout matches LaguerreBackbone exactly:
        Block 1 : OrthoMultiKernelBlock3D  (num_ch → 24, stride=2)
        Block 2 : OrthoVolterraBlock3D     (24 → 32,  stride=2, shortcut)
        Block 3 : OrthoVolterraBlock3D     (32 → 64,  shortcut)
        Block 4 : OrthoVolterraBlock3D     (64 → 96,  stride=2, shortcut)
    Output: [B, 96, T/8, H/8, W/8]
    """

    def __init__(self, basis_kind: str, num_ch: int = 3,
                 N_poly: int | None = None, alpha: float = 1.0):
        super().__init__()
        self.block1 = OrthoMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2, N_poly=N_poly, alpha=alpha, basis_kind=basis_kind,
        )
        self.block2 = OrthoVolterraBlock3D(
            24, 32, Q=4, kernel_size=3, stride=2, use_shortcut=True,
            N_poly=N_poly, alpha=alpha, basis_kind=basis_kind)
        self.block3 = OrthoVolterraBlock3D(
            32, 64, Q=4, kernel_size=3, use_shortcut=True,
            N_poly=N_poly, alpha=alpha, basis_kind=basis_kind)
        self.block4 = OrthoVolterraBlock3D(
            64, 96, Q=4, kernel_size=3, stride=2, use_shortcut=True,
            N_poly=N_poly, alpha=alpha, basis_kind=basis_kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class OrthoHead(nn.Module):
    def __init__(self, num_classes: int, basis_kind: str, num_ch: int = 96,
                 N_poly: int | None = None, alpha: float = 1.0, clip_len: int = 16):
        super().__init__()
        self.block = OrthoVolterraBlock3D(
            num_ch, 256, Q=2, kernel_size=3,
            N_poly=N_poly, alpha=alpha, basis_kind=basis_kind,
            stride=2, use_shortcut=True,
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


class OrthoRgb(nn.Module):
    """Single-stream RGB model with orthogonal basis temporal convolutions."""

    def __init__(self, basis_kind: str, num_classes: int,
                 N_poly: int | None = None, alpha: float = 1.0, clip_len: int = 16):
        super().__init__()
        self.backbone = OrthoBackbone(basis_kind, num_ch=3, N_poly=N_poly, alpha=alpha)
        self.head     = OrthoHead(num_classes, basis_kind, N_poly=N_poly, alpha=alpha,
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


class OrthoFusion(nn.Module):
    """Two-stream (RGB + optical flow) fusion with orthogonal basis convolutions.

    Additive fusion: cat(rgb, flow) only — no explicit cross-stream product.
    """

    def __init__(self, basis_kind: str, num_classes: int,
                 N_poly: int | None = None, alpha: float = 1.0, clip_len: int = 16):
        super().__init__()
        self.model_rgb = OrthoBackbone(basis_kind, num_ch=3, N_poly=N_poly, alpha=alpha)
        self.model_of  = OrthoBackbone(basis_kind, num_ch=2, N_poly=N_poly, alpha=alpha)
        self.head      = OrthoHead(num_classes, basis_kind, num_ch=192,
                                   N_poly=N_poly, alpha=alpha, clip_len=clip_len)

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of  = self.model_of(flow)
        return self.head(torch.cat((out_rgb, out_of), dim=1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.head.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.head.classifier.fc.parameters():
            if p.requires_grad:
                yield p


# ---------------------------------------------------------------------------
# Named constructors for model_factory
# ---------------------------------------------------------------------------

def lvn_legendre_rgb(num_classes: int, clip_len: int = 16,
                     n_poly: int | None = None, alpha: float = 1.0) -> OrthoRgb:
    return OrthoRgb('legendre', num_classes, N_poly=n_poly, alpha=alpha, clip_len=clip_len)

def lvn_chebyshev_rgb(num_classes: int, clip_len: int = 16,
                      n_poly: int | None = None, alpha: float = 1.0) -> OrthoRgb:
    return OrthoRgb('chebyshev', num_classes, N_poly=n_poly, alpha=alpha, clip_len=clip_len)

def lvn_hermite_rgb(num_classes: int, clip_len: int = 16,
                    n_poly: int | None = None, alpha: float = 1.0) -> OrthoRgb:
    return OrthoRgb('hermite', num_classes, N_poly=n_poly, alpha=alpha, clip_len=clip_len)


# ---------------------------------------------------------------------------
# Full 3D orthogonal basis — Tucker decomposition over T, H, W
# ---------------------------------------------------------------------------

class OrthoConv3d_Full(nn.Module):
    """Conv3d with orthogonal polynomial basis along T, H, and W (Tucker decomposition).

    The same polynomial family parameterises all three kernel dimensions.
    Kernel reconstructed at each forward pass:

        W[o,i,t,h,w] = Σ_{n,m,k} coeff[o,i,n,m,k] · basis_T[n,t] · basis_H[m,h] · basis_W[k,w]

    N_poly_T and N_poly_S are capped at the kernel extent to avoid a linearly
    dependent (overcomplete) basis.

    Args:
        in_ch, out_ch  : Channel dimensions.
        kernel_size    : (T, H, W) or int.
        N_poly_T       : Basis functions along T. None → T (no compression).
        N_poly_S       : Basis functions along H and W (shared). None → no compression.
        alpha          : Scale for Laguerre/Hermite; ignored for Legendre/Chebyshev.
        basis_kind     : Polynomial family.
        padding, bias  : Passed to F.conv3d.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size,
                 N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0, basis_kind: str = 'legendre',
                 padding: int | tuple[int, ...] = 0, bias: bool = True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        T, H, W = kernel_size

        N_T = min(N_poly_T, T) if N_poly_T is not None else T
        N_H = min(N_poly_S, H) if N_poly_S is not None else H
        N_W = min(N_poly_S, W) if N_poly_S is not None else W

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding

        self.register_buffer("basis_T", compute_ortho_basis(T, N_T, basis_kind, alpha))
        self.register_buffer("basis_H", compute_ortho_basis(H, N_H, basis_kind, alpha))
        self.register_buffer("basis_W", compute_ortho_basis(W, N_W, basis_kind, alpha))

        self.coeff = nn.Parameter(torch.empty(out_ch, in_ch, N_T, N_H, N_W))
        self.bias_param = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self._init_weights()

    def _init_weights(self):
        T, H, W = self.kernel_size
        std = math.sqrt(2.0 / (self.in_ch * T * H * W))
        nn.init.normal_(self.coeff, std=std)

    def _get_kernel(self) -> torch.Tensor:
        k = torch.einsum("oinmk,nt->oitmk", self.coeff, self.basis_T)
        k = torch.einsum("oitmk,mh->oithk", k,          self.basis_H)
        k = torch.einsum("oithk,kw->oithw", k,          self.basis_W)
        return k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self._get_kernel(), self.bias_param, padding=self.padding)


class OrthoFullVolterraBlock3D(nn.Module):
    """Volterra block using OrthoConv3d_Full (T+H+W basis)."""

    def __init__(self, in_ch: int, out_ch: int, Q: int = 4,
                 kernel_size=3, N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0, basis_kind: str = 'legendre',
                 stride: int = 1, use_shortcut: bool = False):
        super().__init__()
        self.out_ch = out_ch
        self.Q = Q
        self.use_shortcut = use_shortcut

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        pad = tuple(k // 2 for k in kernel_size)

        def _make(out_channels: int) -> OrthoConv3d_Full:
            return OrthoConv3d_Full(in_ch, out_channels, kernel_size,
                                    N_poly_T=N_poly_T, N_poly_S=N_poly_S,
                                    alpha=alpha, basis_kind=basis_kind, padding=pad)

        self.conv_lin  = _make(out_ch)
        self.bn_lin    = nn.BatchNorm3d(out_ch)
        self.conv_quad = _make(2 * Q * out_ch)
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


class OrthoFullMultiKernelBlock3D(nn.Module):
    """Multi-scale first block using OrthoConv3d_Full."""

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 Q: int = 4, N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0, basis_kind: str = 'legendre', stride: int = 1):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)

        self.lin_convs:  nn.ModuleList = nn.ModuleList()
        self.quad_convs: nn.ModuleList = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            kw = dict(N_poly_T=N_poly_T, N_poly_S=N_poly_S,
                      alpha=alpha, basis_kind=basis_kind, padding=pad)
            self.lin_convs.append(
                OrthoConv3d_Full(in_ch, ch_per_kernel, ks, **kw))
            self.quad_convs.append(
                OrthoConv3d_Full(in_ch, 2 * Q * ch_per_kernel, ks, **kw))

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
            [volterra_quadratic(c(x), self.Q, self.ch_per_kernel)
             for c in self.quad_convs], dim=1,
        ))
        return self.pool(lin + self.quad_gate.view(1, -1, 1, 1, 1) * quad)


class OrthoFullBackbone(nn.Module):
    """4-block backbone using OrthoConv3d_Full (T+H+W basis).
    Output: [B, 96, T/8, H/8, W/8]
    """

    def __init__(self, basis_kind: str, num_ch: int = 3,
                 N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0):
        super().__init__()
        kw = dict(N_poly_T=N_poly_T, N_poly_S=N_poly_S, alpha=alpha, basis_kind=basis_kind)
        self.block1 = OrthoFullMultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2, **kw,
        )
        self.block2 = OrthoFullVolterraBlock3D(24, 32, Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)
        self.block3 = OrthoFullVolterraBlock3D(32, 64, Q=4, kernel_size=3,           use_shortcut=True, **kw)
        self.block4 = OrthoFullVolterraBlock3D(64, 96, Q=4, kernel_size=3, stride=2, use_shortcut=True, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block4(self.block3(self.block2(self.block1(x))))


class OrthoFullHead(nn.Module):
    def __init__(self, num_classes: int, basis_kind: str, num_ch: int = 96,
                 N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0, clip_len: int = 16):
        super().__init__()
        self.block = OrthoFullVolterraBlock3D(
            num_ch, 256, Q=2, kernel_size=3, stride=2, use_shortcut=True,
            N_poly_T=N_poly_T, N_poly_S=N_poly_S, alpha=alpha, basis_kind=basis_kind,
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


class OrthoFullFusion(nn.Module):
    """Two-stream fusion with full 3D orthogonal basis. Additive: cat(rgb, flow) → 192ch."""

    def __init__(self, basis_kind: str, num_classes: int,
                 N_poly_T: int | None = None, N_poly_S: int | None = None,
                 alpha: float = 1.0, clip_len: int = 16):
        super().__init__()
        kw = dict(N_poly_T=N_poly_T, N_poly_S=N_poly_S, alpha=alpha)
        self.model_rgb = OrthoFullBackbone(basis_kind, num_ch=3, **kw)
        self.model_of  = OrthoFullBackbone(basis_kind, num_ch=2, **kw)
        self.head      = OrthoFullHead(num_classes, basis_kind, num_ch=192,
                                       clip_len=clip_len, **kw)

    def forward(self, x):
        rgb, flow = x
        return self.head(torch.cat((self.model_rgb(rgb), self.model_of(flow)), dim=1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.head.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.head.classifier.fc.parameters():
            if p.requires_grad:
                yield p


# ---------------------------------------------------------------------------
# Named constructors
# ---------------------------------------------------------------------------

def lvn_legendre_fusion(num_classes: int, clip_len: int = 16,
                        n_poly: int | None = None, n_poly_s: int | None = None,
                        alpha: float = 1.0) -> OrthoFullFusion:
    return OrthoFullFusion('legendre', num_classes, N_poly_T=n_poly, N_poly_S=n_poly_s,
                           alpha=alpha, clip_len=clip_len)

def lvn_chebyshev_fusion(num_classes: int, clip_len: int = 16,
                         n_poly: int | None = None, n_poly_s: int | None = None,
                         alpha: float = 1.0) -> OrthoFullFusion:
    return OrthoFullFusion('chebyshev', num_classes, N_poly_T=n_poly, N_poly_S=n_poly_s,
                           alpha=alpha, clip_len=clip_len)

def lvn_hermite_fusion(num_classes: int, clip_len: int = 16,
                       n_poly: int | None = None, n_poly_s: int | None = None,
                       alpha: float = 1.0) -> OrthoFullFusion:
    return OrthoFullFusion('hermite', num_classes, N_poly_T=n_poly, N_poly_S=n_poly_s,
                           alpha=alpha, clip_len=clip_len)
