"""
1D VNN model for time series / sequential data classification.

Architecture mirrors the 4-block video backbone (network/video_higher_order/vnn_4block.py)
with Conv3d replaced by Conv1d and spatial dimensions removed.

Classes:
    Backbone1D — 4-block 1D VNN backbone, 8× temporal downsampling
    VNN1D      — End-to-end classifier (Backbone1D + global avg pool + FC)

Channel width is controlled by ``base_ch`` (default 8):
    Block 1: MultiKernel  → 3 * base_ch
    Block 2: Quadratic    → 4 * base_ch
    Block 3: Quad+Cubic   → 8 * base_ch
    Block 4: Quad+Cubic   → 12 * base_ch
    GAP → FC(12 * base_ch, num_classes)

Rough param counts (Q=2, Qc=1, symmetric cubic):
    base_ch=4  →  ~50K
    base_ch=8  →  ~200K
    base_ch=16 →  ~800K
"""

import torch.nn as nn

from .volterra_blocks_1d import MultiKernelBlock1D, VolterraBlock1D


class Backbone1D(nn.Module):
    """4-block 1D VNN backbone.

    Architecture:
        Block 1: Multi-kernel (k=9,5,1) → Quadratic → Pool (/2)
        Block 2: Single kernel (k=3)    → Quadratic → Pool (/2)
        Block 3: Single kernel (k=3)    → Quadratic + Cubic (no pool)
        Block 4: Single kernel (k=3)    → Quadratic + Cubic → Pool (/2)

    Total downsampling: 8×  →  output [B, 12*base_ch, seq_len // 8]

    Args:
        in_ch: Input channels (1 = univariate, N = multivariate).
        base_ch: Base channel width. Output channels are 3/4/8/12 × base_ch per block.
        Q: Quadratic rank (default 2).
        Qc: Cubic rank (default 1).
        cubic_mode: ``'symmetric'`` or ``'general'`` cubic factorization.
        use_cubic: Enable 3rd-order cubic paths in blocks 3 and 4.
    """

    def __init__(self, in_ch=1, base_ch=8, Q=2, Qc=1,
                 cubic_mode='symmetric', use_cubic=True):
        super().__init__()
        c1 = 3 * base_ch   # block1 out
        c2 = 4 * base_ch   # block2 out
        c3 = 8 * base_ch   # block3 out
        c4 = 12 * base_ch  # block4 out

        self.out_ch = c4

        self.block1 = MultiKernelBlock1D(
            in_ch, ch_per_kernel=base_ch,
            kernels=[9, 5, 1],
            Q=Q, stride=2,
        )

        self.block2 = VolterraBlock1D(c1, c2, Q=Q, stride=2, use_shortcut=True)

        # Boost Q when cubic is disabled to preserve expressiveness
        cubic_mult = 3 if cubic_mode == 'general' else 2
        q_eff = (Q + cubic_mult * Qc // 2) if not use_cubic else Q

        self.block3 = VolterraBlock1D(
            c2, c3, Q=q_eff, Qc=Qc,
            use_cubic=use_cubic, cubic_mode=cubic_mode,
            use_shortcut=True,
        )

        self.block4 = VolterraBlock1D(
            c3, c4, Q=q_eff, Qc=Qc, stride=2,
            use_cubic=use_cubic, cubic_mode=cubic_mode,
            use_shortcut=True,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class VNN1D(nn.Module):
    """End-to-end 1D VNN for time series classification.

    Architecture::

        [B, in_ch, T]
          → Backbone1D          [B, 12*base_ch, T/8]
          → AdaptiveAvgPool1d   [B, 12*base_ch, 1]
          → Dropout → FC        [B, num_classes]

    AdaptiveAvgPool1d makes the model input-length-agnostic.

    Args:
        num_classes: Number of output classes.
        in_ch: Input channels (1 for univariate, N for N-variate).
        base_ch: Base channel width (default 8, use 4 for ~50K params).
        Q: Quadratic rank (default 2).
        Qc: Cubic rank (default 1).
        cubic_mode: ``'symmetric'`` or ``'general'`` cubic factorization.
        use_cubic: Enable 3rd-order cubic paths.
        dropout: Dropout probability before the FC layer (default 0.5).
    """

    def __init__(self, num_classes, in_ch=1, base_ch=8, Q=2, Qc=1,
                 cubic_mode='symmetric', use_cubic=True, dropout=0.5):
        super().__init__()
        self.backbone = Backbone1D(
            in_ch=in_ch, base_ch=base_ch, Q=Q, Qc=Qc,
            cubic_mode=cubic_mode, use_cubic=use_cubic,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.backbone.out_ch, num_classes)

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
