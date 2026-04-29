"""
Simplified variants of LaguerreVNN1D for ablation studies.

Three separate model types, each applying one simplification to the base
LaguerreVNN1D architecture.  All other hyperparameters (channel widths,
block topology, kernel sizes) are identical to LaguerreVNN1D so results
are directly comparable.

S1 — No inner clamp (laguerre_vnn_1d_s1):
    Removes the clamp(-20, 20) on the softplus argument inside laguerre_feature.
    Original: softplus(clamp(z, -20, 20) * alpha)
    S1:       softplus(z * alpha)
    Rationale: block-input clamping (±50) plus BN normalization already bound z
    in practice.  The inner clamp adds a stop-gradient discontinuity with no
    benefit for well-conditioned inputs.

S2 — Shared projection across degrees (laguerre_vnn_1d_s2):
    Replaces the per-degree Conv1d with a single shared Conv1d; each degree
    gets its own learnable scale alpha_d (initialized to alpha) so it can
    still adjust its polynomial range independently.
    Original: phi_d = BN(L_d(softplus(alpha * W_d * x)))   # W_d separate per d
    S2:       phi_d = BN(L_d(softplus(alpha_d * W_shared * x)))  # W_shared shared
    Rationale: cuts (|degrees| - 1) * base_conv_params from every block.
    At [2,3] degrees and base_ch=8 this saves ~30% of poly-path parameters.

S3 — Scalar gates (laguerre_vnn_1d_s3):
    Replaces the per-channel gate g_d ∈ R^{C_out} with a scalar g_d ∈ R.
    Original: out += gate_d.view(1, C, 1) * phi_d   # gate_d shape (C_out,)
    S3:       out += gate_d * phi_d                  # gate_d shape (1,)
    Rationale: saves |degrees| * (C_out - 1) parameters per block.  The gate
    still controls the overall contribution of each polynomial degree; the per-
    channel modulation is absorbed into the subsequent BN and linear path.
"""

import torch.nn as nn

from .laguerre_poly_blocks_1d import LaguerrePolyBlock1D, MultiKernelLaguerreBlock1D


# ---------------------------------------------------------------------------
# Shared backbone factory
# ---------------------------------------------------------------------------

class _SimplifiedBackbone1D(nn.Module):
    """4-block Laguerre backbone with configurable simplification flags.

    Same topology as LaguerreBackbone1D:
        Block 1: MultiKernel (k=9,5,1) → Pool (/2)
        Block 2: Single (k=3) + shortcut → Pool (/2)
        Block 3: Single (k=3) + shortcut
        Block 4: Single (k=3) + shortcut → Pool (/2)
    Total downsampling: 8×. Output: [B, 12*base_ch, T/8].
    """

    def __init__(self, in_ch: int, base_ch: int, poly_degrees, alpha: float,
                 use_inner_clamp: bool, shared_proj: bool, scalar_gates: bool):
        super().__init__()
        c1 = 3 * base_ch
        c2 = 4 * base_ch
        c3 = 8 * base_ch
        c4 = 12 * base_ch
        self.out_ch = c4

        self.block1 = MultiKernelLaguerreBlock1D(
            in_ch, ch_per_kernel=base_ch, kernels=[9, 5, 1], stride=2,
            poly_degrees=poly_degrees, alpha=alpha,
            use_inner_clamp=use_inner_clamp, shared_proj=shared_proj, scalar_gates=scalar_gates,
        )
        self.block2 = LaguerrePolyBlock1D(
            c1, c2, stride=2, use_shortcut=True,
            poly_degrees=poly_degrees, alpha=alpha,
            use_inner_clamp=use_inner_clamp, shared_proj=shared_proj, scalar_gates=scalar_gates,
        )
        self.block3 = LaguerrePolyBlock1D(
            c2, c3, use_shortcut=True,
            poly_degrees=poly_degrees, alpha=alpha,
            use_inner_clamp=use_inner_clamp, shared_proj=shared_proj, scalar_gates=scalar_gates,
        )
        self.block4 = LaguerrePolyBlock1D(
            c3, c4, stride=2, use_shortcut=True,
            poly_degrees=poly_degrees, alpha=alpha,
            use_inner_clamp=use_inner_clamp, shared_proj=shared_proj, scalar_gates=scalar_gates,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class _SimplifiedVNN1DBase(nn.Module):
    """Base class for simplified LaguerreVNN1D variants."""

    def __init__(self, num_classes: int, in_ch: int, base_ch: int,
                 poly_degrees, alpha: float, dropout: float,
                 use_inner_clamp: bool, shared_proj: bool, scalar_gates: bool):
        super().__init__()
        self.backbone = _SimplifiedBackbone1D(
            in_ch=in_ch, base_ch=base_ch, poly_degrees=poly_degrees, alpha=alpha,
            use_inner_clamp=use_inner_clamp, shared_proj=shared_proj,
            scalar_gates=scalar_gates,
        )
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(self.backbone.out_ch, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

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
# S1: No inner clamp
# ---------------------------------------------------------------------------

class LaguerreVNN1D_S1(_SimplifiedVNN1DBase):
    """LaguerreVNN1D without the inner softplus-argument clamp.

    Removes clamp(z, -20, 20) inside laguerre_feature.  All other
    architecture details are identical to LaguerreVNN1D.

    Args:
        num_classes:  Number of output classes.
        in_ch:        Input channels.
        base_ch:      Base channel width (default 8).
        poly_degrees: Laguerre polynomial degrees.  Default [2, 3].
        alpha:        Softplus scale.
        dropout:      Dropout before FC (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees if poly_degrees is not None else [2, 3],
            alpha=alpha, dropout=dropout,
            use_inner_clamp=False, shared_proj=False, scalar_gates=False,
        )


# ---------------------------------------------------------------------------
# S2: Shared projection across degrees
# ---------------------------------------------------------------------------

class LaguerreVNN1D_S2(_SimplifiedVNN1DBase):
    """LaguerreVNN1D with a shared Conv1d projection across polynomial degrees.

    Replaces the per-degree convolutions with a single shared Conv1d per block;
    each degree gets its own learnable scale alpha_d initialized to alpha.

    Args:
        num_classes:  Number of output classes.
        in_ch:        Input channels.
        base_ch:      Base channel width (default 8).
        poly_degrees: Laguerre polynomial degrees.  Default [2, 3].
        alpha:        Initial value for the per-degree learnable scales alpha_d.
        dropout:      Dropout before FC (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees if poly_degrees is not None else [2, 3],
            alpha=alpha, dropout=dropout,
            use_inner_clamp=True, shared_proj=True, scalar_gates=False,
        )


# ---------------------------------------------------------------------------
# S3: Scalar gates
# ---------------------------------------------------------------------------

class LaguerreVNN1D_S3(_SimplifiedVNN1DBase):
    """LaguerreVNN1D with scalar gates instead of per-channel gates.

    Replaces gate_d ∈ R^{C_out} with gate_d ∈ R (one scalar per degree per
    block).  All other architecture details are identical to LaguerreVNN1D.

    Args:
        num_classes:  Number of output classes.
        in_ch:        Input channels.
        base_ch:      Base channel width (default 8).
        poly_degrees: Laguerre polynomial degrees.  Default [2, 3].
        alpha:        Softplus scale.
        dropout:      Dropout before FC (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees if poly_degrees is not None else [2, 3],
            alpha=alpha, dropout=dropout,
            use_inner_clamp=True, shared_proj=False, scalar_gates=True,
        )


# ---------------------------------------------------------------------------
# S4: Shared projection + no inner clamp  (S2 ∩ S1)
# ---------------------------------------------------------------------------

class LaguerreVNN1D_S4(_SimplifiedVNN1DBase):
    """LaguerreVNN1D with shared projection and no inner clamp.

    Combines S2 (shared Conv1d across degrees) and S1 (no inner clamp).

    Args:
        num_classes:  Number of output classes.
        in_ch:        Input channels.
        base_ch:      Base channel width (default 8).
        poly_degrees: Laguerre polynomial degrees.  Default [2, 3].
        alpha:        Initial value for the per-degree learnable scales alpha_d.
        dropout:      Dropout before FC (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees if poly_degrees is not None else [2, 3],
            alpha=alpha, dropout=dropout,
            use_inner_clamp=False, shared_proj=True, scalar_gates=False,
        )


# ---------------------------------------------------------------------------
# S5: Scalar gates + no inner clamp  (S3 ∩ S1)
# ---------------------------------------------------------------------------

class LaguerreVNN1D_S5(_SimplifiedVNN1DBase):
    """LaguerreVNN1D with scalar gates and no inner clamp.

    Combines S3 (scalar gates) and S1 (no inner clamp).

    Args:
        num_classes:  Number of output classes.
        in_ch:        Input channels.
        base_ch:      Base channel width (default 8).
        poly_degrees: Laguerre polynomial degrees.  Default [2, 3].
        alpha:        Softplus scale.
        dropout:      Dropout before FC (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__(
            num_classes=num_classes, in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees if poly_degrees is not None else [2, 3],
            alpha=alpha, dropout=dropout,
            use_inner_clamp=False, shared_proj=False, scalar_gates=True,
        )
