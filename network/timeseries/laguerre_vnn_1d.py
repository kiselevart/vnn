"""
1D Laguerre polynomial VNN for time series classification.

Drop-in replacement for VNN1D (vnn_1d.py).  Channel widths and block topology
are identical, so models are directly comparable at the same parameter budget.

The only architectural change is that the monomial Volterra interaction paths
(quadratic CP product, cubic a²·b) are replaced by orthogonal Laguerre
polynomial evaluations:

    VNN1D path:
        quad  = L_left * L_right                  (CP decomposition, monomial)
        cubic = a² * b

    LaguerreVNN1D path:
        Σ_{d in poly_degrees} gate_d · BN( L_d(softplus(α·conv_d(x))) )

poly_degrees controls which orders are active:
    [2]       →  quadratic-only  (1 nonlinear path)
    [2, 3]    →  quad + cubic    (default; same count as VNN1D use_cubic=True)
    [2, 3, 4] →  up to quartic   (3 nonlinear paths)

Param count at base_ch=8, poly_degrees=[2,3]:
    ≈ same as VNN1D base_ch=8, use_cubic=True  (~200K)

Classes:
    LaguerreBackbone1D  — 4-block backbone, 8× downsampling
    LaguerreVNN1D       — end-to-end classifier
"""

import torch.nn as nn

from .laguerre_poly_blocks_1d import LaguerrePolyBlock1D, MultiKernelLaguerreBlock1D


class LaguerreBackbone1D(nn.Module):
    """4-block Laguerre polynomial backbone for time series.

    Identical block topology to Backbone1D:
        Block 1: MultiKernel (k=9,5,1) + Laguerre poly → Pool (/2)
        Block 2: Single (k=3)           + Laguerre poly → Pool (/2)
        Block 3: Single (k=3)           + Laguerre poly
        Block 4: Single (k=3)           + Laguerre poly → Pool (/2)

    Total downsampling: 8×. Output: [B, 12*base_ch, T/8].

    Args:
        in_ch:         Input channels.
        base_ch:       Base channel width (default 8).
        poly_degrees:  Laguerre polynomial degrees, e.g. [2, 3].
                       Int N → degrees [2 .. N+1].
        alpha:         Softplus input scale (lower = tighter polynomial range).
    """

    def __init__(self, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0):
        super().__init__()
        if poly_degrees is None:
            poly_degrees = [2, 3]

        c1 = 3 * base_ch
        c2 = 4 * base_ch
        c3 = 8 * base_ch
        c4 = 12 * base_ch
        self.out_ch = c4

        self.block1 = MultiKernelLaguerreBlock1D(
            in_ch, ch_per_kernel=base_ch, kernels=[9, 5, 1], stride=2,
            poly_degrees=poly_degrees, alpha=alpha,
        )
        self.block2 = LaguerrePolyBlock1D(
            c1, c2, stride=2, use_shortcut=True, poly_degrees=poly_degrees, alpha=alpha,
        )
        self.block3 = LaguerrePolyBlock1D(
            c2, c3, use_shortcut=True, poly_degrees=poly_degrees, alpha=alpha,
        )
        self.block4 = LaguerrePolyBlock1D(
            c3, c4, stride=2, use_shortcut=True, poly_degrees=poly_degrees, alpha=alpha,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class LaguerreVNN1D(nn.Module):
    """End-to-end Laguerre polynomial VNN for time series classification.

    Architecture::

        [B, in_ch, T]
          → LaguerreBackbone1D     [B, 12*base_ch, T/8]
          → AdaptiveAvgPool1d      [B, 12*base_ch, 1]
          → Dropout → FC           [B, num_classes]

    AdaptiveAvgPool1d makes the model input-length-agnostic.

    Args:
        num_classes:   Number of output classes.
        in_ch:         Input channels (1 for univariate, N for N-variate).
        base_ch:       Base channel width (default 8, use 4 for ~50K params).
        poly_degrees:  Laguerre degrees or int N.  Default [2, 3].
        alpha:         Softplus scale.  Try 0.5 when using degrees ≥ 4 to
                       keep polynomial inputs in a stable range.
        dropout:       Dropout probability before FC layer (default 0.5).
    """

    def __init__(self, num_classes: int, in_ch: int = 1, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 1.0, dropout: float = 0.5):
        super().__init__()
        self.backbone = LaguerreBackbone1D(
            in_ch=in_ch, base_ch=base_ch,
            poly_degrees=poly_degrees, alpha=alpha,
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
        """All parameters except the final FC layer."""
        skip = {id(p) for p in self.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        """Final FC layer parameters (for higher learning rate)."""
        for p in self.fc.parameters():
            if p.requires_grad:
                yield p
