"""
1D Laguerre polynomial interaction blocks for time series classification.

Replaces monomial Volterra interactions (CP products: left*right, a²*b) with
a tunable sum of orthogonal Laguerre polynomial evaluations:

    out = BN(conv_lin(x))
        + Σ_{d in poly_degrees}  gate_d · BN( L_d(softplus(α·conv_d(x))) )

Laguerre polynomials L_n are orthogonal on [0, ∞) under the e^{-t} measure:
    L_0(t) = 1
    L_1(t) = 1 - t
    L_n(t) = ((2n-1-t)·L_{n-1}(t) - (n-1)·L_{n-2}(t)) / n

softplus maps conv outputs from R to (0, ∞), placing them in the valid Laguerre
domain. Each degree gets a separate convolution (different linear projection of
the input) and a small gated contribution (init 1e-4), so the model starts
near-linear and gradually engages higher-order polynomial terms.

Increasing poly_degrees adds orthogonal nonlinear modes with independent gradient
paths — unlike monomial powers, where h1·∂(x²)/∂x = 2h1·x mixes first and
second-order gradients.

Classes:
    LaguerrePolyBlock1D        — single-kernel block
    MultiKernelLaguerreBlock1D — multi-scale first block
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Laguerre polynomial evaluation
# ---------------------------------------------------------------------------

def laguerre_poly(n: int, t: torch.Tensor) -> torch.Tensor:
    """Evaluate L_n(t) elementwise via three-term recurrence.

    Valid for any real t, though orthogonality holds only on [0, ∞).
    """
    if n == 0:
        return torch.ones_like(t)
    L2 = torch.ones_like(t)   # L_{k-2} (starts as L_0)
    L1 = 1.0 - t              # L_{k-1} (starts as L_1)
    if n == 1:
        return L1
    for k in range(2, n + 1):
        L0 = ((2 * k - 1 - t) * L1 - (k - 1) * L2) / k
        L2, L1 = L1, L0
    return L1


def laguerre_feature(z: torch.Tensor, degree: int, alpha: float = 1.0) -> torch.Tensor:
    """Compute L_degree(softplus(α·z)), clamped to [-50, 50].

    softplus(·) maps z ∈ R to (0, ∞), placing features in the Laguerre domain.
    The clamp guards against polynomial growth at large t for high degrees.
    """
    t = F.softplus(z.clamp(-20.0, 20.0) * alpha)   # (0, ∞)
    return laguerre_poly(degree, t).clamp(-50.0, 50.0)


def _parse_degrees(poly_degrees) -> list:
    """Normalise poly_degrees to a list.

    int N → degrees [2, 3, ..., N+1]  (N degrees, starting at quadratic)
    list  → used as-is
    """
    if isinstance(poly_degrees, int):
        return list(range(2, poly_degrees + 2))
    return list(poly_degrees)


def _init_1d(module):
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# ---------------------------------------------------------------------------
# LaguerrePolyBlock1D
# ---------------------------------------------------------------------------

class LaguerrePolyBlock1D(nn.Module):
    """Single-kernel 1D block with Laguerre polynomial interactions.

    Architecture::

        x = clamp(x)
        out = BN(conv_lin(x))
            + Σ_d  gate_d · BN( L_d(softplus(α · conv_d(x))) )
        out = [shortcut(x) +] out
        out = pool(out)

    Each polynomial degree d uses its own Conv1d (independent projection of x)
    and a separate learnable gate (init 1e-4).  This mirrors having separate
    quadratic / cubic convolution paths in VolterraBlock1D, generalised to an
    arbitrary list of polynomial degrees.

    Args:
        in_ch:         Input channels.
        out_ch:        Output channels.
        poly_degrees:  List of Laguerre degrees, e.g. [2, 3].
                       Or an int N → degrees [2 .. N+1].
                       Default [2, 3] matches quad+cubic Volterra block.
        stride:        If > 1, applies MaxPool1d(2) after summation.
        use_shortcut:  Add residual 1×1 conv shortcut.
        kernel_size:   Temporal convolution kernel size.
        alpha:         Scale applied to conv output before softplus.
                       Lower alpha keeps inputs in a tighter polynomial range,
                       which helps stability for degree ≥ 4.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 poly_degrees=None, stride: int = 1,
                 use_shortcut: bool = False, kernel_size: int = 3,
                 alpha: float = 1.0):
        super().__init__()
        if poly_degrees is None:
            poly_degrees = [2, 3]
        self.poly_degrees = _parse_degrees(poly_degrees)
        self.out_ch       = out_ch
        self.use_shortcut = use_shortcut
        self.alpha        = alpha

        pad = kernel_size // 2

        self.conv_lin = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn_lin   = nn.BatchNorm1d(out_ch)

        self.poly_convs = nn.ModuleList([
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
            for _ in self.poly_degrees
        ])
        self.poly_bns = nn.ModuleList([
            nn.BatchNorm1d(out_ch) for _ in self.poly_degrees
        ])
        self.poly_gates = nn.ParameterList([
            nn.Parameter(torch.ones(out_ch) * 1e-4)
            for _ in self.poly_degrees
        ])

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        self.pool = nn.MaxPool1d(2, 2) if stride > 1 else nn.Identity()
        _init_1d(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clamp(-50.0, 50.0)
        out = self.bn_lin(self.conv_lin(x))

        for conv, bn, gate, deg in zip(
            self.poly_convs, self.poly_bns, self.poly_gates, self.poly_degrees
        ):
            phi = bn(laguerre_feature(conv(x), deg, self.alpha))
            out = out + gate.view(1, -1, 1) * phi

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


# ---------------------------------------------------------------------------
# MultiKernelLaguerreBlock1D
# ---------------------------------------------------------------------------

class MultiKernelLaguerreBlock1D(nn.Module):
    """Multi-kernel first block with Laguerre polynomial interactions.

    Parallel branches at different temporal scales (kernel sizes).  Within each
    branch, a linear path and one Laguerre path per degree are computed; all
    branches are concatenated before BN.

    Total output channels = ch_per_kernel × len(kernels).

    Args:
        in_ch:          Input channels.
        ch_per_kernel:  Output channels per kernel branch.
        kernels:        List of kernel sizes, e.g. [9, 5, 1].
        poly_degrees:   Laguerre degrees or int N.  Default [2, 3].
        stride:         If > 1, MaxPool1d(2) applied after summation.
        use_shortcut:   Residual 1×1 shortcut.
        alpha:          Softplus input scale.
    """

    def __init__(self, in_ch: int, ch_per_kernel: int, kernels,
                 poly_degrees=None, stride: int = 1,
                 use_shortcut: bool = False, alpha: float = 1.0):
        super().__init__()
        if poly_degrees is None:
            poly_degrees = [2, 3]
        self.poly_degrees  = _parse_degrees(poly_degrees)
        self.ch_per_kernel = ch_per_kernel
        self.out_ch        = ch_per_kernel * len(kernels)
        self.use_shortcut  = use_shortcut
        self.alpha         = alpha

        self.lin_convs = nn.ModuleList([
            nn.Conv1d(in_ch, ch_per_kernel, ks, padding=ks // 2)
            for ks in kernels
        ])
        self.bn_lin = nn.BatchNorm1d(self.out_ch)

        # poly_convs[degree_idx] is a ModuleList over kernel sizes
        self.poly_convs = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(in_ch, ch_per_kernel, ks, padding=ks // 2)
                for ks in kernels
            ])
            for _ in self.poly_degrees
        ])
        self.poly_bns = nn.ModuleList([
            nn.BatchNorm1d(self.out_ch) for _ in self.poly_degrees
        ])
        self.poly_gates = nn.ParameterList([
            nn.Parameter(torch.ones(self.out_ch) * 1e-4)
            for _ in self.poly_degrees
        ])

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm1d(self.out_ch),
            )

        self.pool = nn.MaxPool1d(2, 2) if stride > 1 else nn.Identity()
        _init_1d(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clamp(-50.0, 50.0)
        out = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))

        for deg_convs, bn, gate, deg in zip(
            self.poly_convs, self.poly_bns, self.poly_gates, self.poly_degrees
        ):
            phis = torch.cat([
                laguerre_feature(c(x), deg, self.alpha)
                for c in cast(nn.ModuleList, deg_convs)
            ], dim=1)
            out = out + gate.view(1, -1, 1) * bn(phis)

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)
