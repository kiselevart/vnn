"""
Volterra math primitives and reusable 3D building blocks for video models.

Volterra series expansion up to 3rd order:
    y = h0 + Σ h1(i)·x(i) + Σ h2(i,j)·x(i)·x(j) + Σ h3(i,j,k)·x(i)·x(j)·x(k)

Decomposition strategies:
    2nd-order (quadratic):  CP factorization    → left · right   (2·Q·C channels)
    3rd-order symmetric:    Tied CP factors      → a² · b         (2·Q·C channels)
    3rd-order general:      Independent factors  → a · b · c      (3·Q·C channels)

Building blocks:
    VolterraBlock3D    — Single-kernel block (linear + quadratic + optional cubic)
    MultiKernelBlock3D — Multi-kernel block (parallel convs at different scales)
    ClassifierHead     — Flatten → Dropout → Linear classifier
"""

import torch
import torch.nn as nn

# Factor clamp values: chosen so a single un-summed product term is ≤ 50 in magnitude.
# Clamping factors (not outputs) bounds the backward gradient through the product:
#   quadratic:  ∂(left*right)/∂left  = right  → bounded by _QUAD_FC  = √50 ≈ 7.07
#   cubic sym:  ∂(a²*b)/∂a           = 2ab    → bounded by 2·_CUB_FC² ≈ 27
#   cubic gen:  ∂(a*b*c)/∂a          = b*c    → bounded by _CUB_FC²  ≈ 13.5
# The post-sum output clamp (±50) stays as a backstop for the Q-fold summation.
_QUAD_FC = 7.071   # √50
_CUB_FC  = 3.684   # ∛50


# ---------------------------------------------------------------------------
# Volterra interaction primitives
# ---------------------------------------------------------------------------

def volterra_quadratic(x_conv, Q, nch_out):
    """Vectorized 2nd-order Volterra interaction (CP decomposition).

    h2(i,j) ≈ Σ_q a_q(i)·b_q(j)
    Splits channels into left/right halves and element-wise multiplies.

    Args:
        x_conv: Tensor [B, 2*Q*C, T, H, W] from quadratic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    mid = Q * nch_out
    # Clamp factors before multiplying — bounds backward gradient to ≤ _QUAD_FC per factor.
    left  = x_conv[:, :mid].clamp(-_QUAD_FC, _QUAD_FC)
    right = x_conv[:, mid:].clamp(-_QUAD_FC, _QUAD_FC)

    product = left * right  # [B, Q*C, T, H, W], |product| ≤ 50

    shape = product.shape
    # Post-sum clamp guards against Q-fold accumulation exceeding ±50.
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


def volterra_cubic_symmetric(x_conv, Q, nch_out):
    """Symmetric 3rd-order Volterra interaction (a²·b decomposition).

    h3(i,j,k) ≈ Σ_q a_q(i)·a_q(j)·b_q(k)
    Two factors are tied → only 2·Q·C channels needed (same as quadratic).

    The a² term acts as an energy/magnitude detector (always ≥ 0),
    while b modulates sign and scale — a learned nonlinear gating mechanism.

    Args:
        x_conv: Tensor [B, 2*Q*C, T, H, W] from cubic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    mid = Q * nch_out
    # Clamp factors — bounds ∂(a²b)/∂a = 2ab to ≤ 2·_CUB_FC².
    a = x_conv[:, :mid].clamp(-_CUB_FC, _CUB_FC)   # Feature detector (will be squared)
    b = x_conv[:, mid:].clamp(-_CUB_FC, _CUB_FC)   # Modulator/gate

    product = (a * a) * b  # [B, Q*C, T, H, W], |product| ≤ 50

    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


def volterra_cubic_general(x_conv, Q, nch_out):
    """General 3rd-order Volterra interaction (a·b·c decomposition).

    h3(i,j,k) ≈ Σ_q a_q(i)·b_q(j)·c_q(k)
    Three independent factors → 3·Q·C channels needed.
    More expressive than symmetric, but more parameters.

    Args:
        x_conv: Tensor [B, 3*Q*C, T, H, W] from cubic expansion conv.
        Q: Number of interaction rank components.
        nch_out: Output channels C per group.
    Returns:
        Tensor [B, C, T, H, W].
    """
    # Clamp factors — bounds ∂(a*b*c)/∂a = b*c to ≤ _CUB_FC².
    a, b, c = [t.clamp(-_CUB_FC, _CUB_FC) for t in torch.chunk(x_conv, 3, dim=1)]

    product = a * b * c  # [B, Q*C, T, H, W], |product| ≤ 50

    shape = product.shape
    return product.view(shape[0], Q, nch_out, *shape[2:]).sum(dim=1).clamp(-50.0, 50.0)


def init_vnn_weights(module):
    """Standard VNN weight initialization.

    Conv3d: Kaiming normal (fan_in, suitable for Volterra nonlinearity).
    BatchNorm3d: weight=1, bias=0.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class VolterraBlock3D(nn.Module):
    """Single-kernel 3D Volterra block.

    Computes::

        out = [shortcut(x) +] linear(x) + [quad_gate *] quad(x) [+ cubic_gate * cubic(x)]
        out = pool(out)

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        Q: Rank for quadratic decomposition (default 4).
        Qc: Rank for cubic decomposition (default 2).
        stride: If > 1, applies MaxPool3d(2) after summation.
        use_cubic: Enable 3rd-order cubic path.
        cubic_mode: ``'symmetric'`` (a²·b, 2Q channels) or
                    ``'general'`` (a·b·c, 3Q channels).
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Zero-init gate on the quadratic branch.
        kernel_size: Convolution kernel size (int or tuple).
    """

    def __init__(self, in_ch, out_ch, Q=4, Qc=2, stride=1,
                 use_cubic=False, cubic_mode='symmetric',
                 use_shortcut=False, gate_quadratic=True,
                 kernel_size=3):
        super().__init__()
        self.out_ch = out_ch
        self.Q = Q
        self.Qc = Qc
        self.use_cubic = use_cubic
        self.cubic_mode = cubic_mode
        self.use_shortcut = use_shortcut
        self.gate_quadratic = gate_quadratic

        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        pad = tuple(k // 2 for k in ks)

        # --- Linear path ---
        self.conv_lin = nn.Conv3d(in_ch, out_ch, ks, padding=pad)
        self.bn_lin = nn.BatchNorm3d(out_ch)

        # --- Quadratic path ---
        self.conv_quad = nn.Conv3d(in_ch, 2 * Q * out_ch, ks, padding=pad)
        self.bn_quad = nn.BatchNorm3d(out_ch)
        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        # --- Cubic path (optional) ---
        if use_cubic:
            cubic_mult = 3 if cubic_mode == 'general' else 2
            self.conv_cubic = nn.Conv3d(in_ch, cubic_mult * Qc * out_ch, ks, padding=pad)
            self.bn_cubic = nn.BatchNorm3d(out_ch)
            self.cubic_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        # --- Shortcut (optional) ---
        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
            )

        # --- Pooling ---
        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        x = x.clamp(-50.0, 50.0)  # block-level firewall: prevents accumulated BN drift from feeding explosive values into interactions
        # Linear
        out = self.bn_lin(self.conv_lin(x))

        # Quadratic
        q = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
        if self.gate_quadratic:
            q = self.quad_gate.view(1, -1, 1, 1, 1) * q
        out = out + q

        # Cubic
        if self.use_cubic:
            fn = volterra_cubic_general if self.cubic_mode == 'general' else volterra_cubic_symmetric
            c = self.bn_cubic(fn(self.conv_cubic(x), self.Qc, self.out_ch))
            out = out + self.cubic_gate.view(1, -1, 1, 1, 1) * c

        # Shortcut
        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


class MultiKernelBlock3D(nn.Module):
    """Multi-kernel 3D Volterra block with parallel convolutions.

    Each kernel produces ``ch_per_kernel`` channels; outputs are concatenated.
    Total output channels = ``ch_per_kernel × len(kernels)``.

    Computes::

        out = [shortcut(x) +] concat_linear(x) + [quad_gate *] concat_quad(x)
        out = pool(out)

    Args:
        in_ch: Input channels.
        ch_per_kernel: Output channels per kernel branch.
        kernels: List of kernel sizes, e.g. ``[(5,5,5), (3,3,3), (1,1,1)]``.
        Q: Rank for quadratic decomposition.
        stride: If > 1, applies MaxPool3d(2) after summation.
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Zero-init gate on the quadratic branch.
    """

    def __init__(self, in_ch, ch_per_kernel, kernels, Q=4, stride=1,
                 use_shortcut=False, gate_quadratic=True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.kernels = kernels
        self.out_ch = ch_per_kernel * len(kernels)
        self.use_shortcut = use_shortcut
        self.gate_quadratic = gate_quadratic

        # Linear paths (one conv per kernel size)
        self.lin_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.lin_convs.append(nn.Conv3d(in_ch, ch_per_kernel, ks, padding=pad))
        self.bn_lin = nn.BatchNorm3d(self.out_ch)

        # Quadratic paths (one conv per kernel size)
        self.quad_convs = nn.ModuleList()
        for ks in kernels:
            pad = tuple(k // 2 for k in ks)
            self.quad_convs.append(nn.Conv3d(in_ch, 2 * Q * ch_per_kernel, ks, padding=pad))
        self.bn_quad = nn.BatchNorm3d(self.out_ch)

        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm3d(self.out_ch),
            )

        self.pool = nn.MaxPool3d(2, 2) if stride > 1 else nn.Identity()

        init_vnn_weights(self)

    def forward(self, x):
        x = x.clamp(-50.0, 50.0)  # block-level firewall
        # Linear: parallel convs → concat → BN
        lin = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))

        # Quadratic: parallel interactions → concat → BN
        quads = [volterra_quadratic(c(x), self.Q, self.ch_per_kernel)
                 for c in self.quad_convs]
        quad = self.bn_quad(torch.cat(quads, dim=1))

        if self.gate_quadratic:
            quad = self.quad_gate.view(1, -1, 1, 1, 1) * quad

        out = lin + quad

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


class TemporalAttentionPool(nn.Module):
    """Spatial GAP + learned temporal attention pooling.

    1. Spatial average pool over H × W  →  [B, C, T]
    2. Linear scorer: one scalar weight per frame  →  softmax over T
    3. Weighted sum over T  →  [B, C]

    The scorer is a single Linear(C → 1); the softmax ensures weights sum to 1
    so the output magnitude stays in the same range regardless of clip length.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_gap = nn.AdaptiveAvgPool3d((None, 1, 1))  # pool H, W; keep T
        self.scorer = nn.Linear(channels, 1, bias=True)
        nn.init.zeros_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.spatial_gap(x).squeeze(-1).squeeze(-1)  # [B, C, T]
        # scores: [B, T, 1] → softmax over T
        scores = torch.softmax(self.scorer(x.permute(0, 2, 1)), dim=1)  # [B, T, 1]
        # weighted sum: [B, C, T] * [B, 1, T] → [B, C]
        return (x * scores.permute(0, 2, 1)).sum(dim=2)


class ClassifierHead(nn.Module):
    """[B, C] → Dropout → FC classifier.

    Args:
        fc_features: Number of input features (channels after pooling).
        num_classes: Number of output classes.
        dropout: Dropout probability (default 0.5).
    """

    def __init__(self, fc_features, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
