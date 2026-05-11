"""Tiny two-block S2-style Laguerre VNN for MNIST — fully self-contained."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Laguerre math
# ---------------------------------------------------------------------------

def _laguerre_poly(n: int, t: torch.Tensor) -> torch.Tensor:
    """Evaluate L_n(t) elementwise via three-term recurrence.

    L_0 = 1,  L_1 = 1 - t,  L_k = ((2k-1-t)·L_{k-1} - (k-1)·L_{k-2}) / k
    """
    if n == 0:
        return torch.ones_like(t)
    L2 = torch.ones_like(t)
    L1 = 1.0 - t
    if n == 1:
        return L1
    for k in range(2, n + 1):
        L0 = ((2 * k - 1 - t) * L1 - (k - 1) * L2) / k
        L2, L1 = L1, L0
    return L1


def _laguerre_feature(z: torch.Tensor, degree: int, alpha: float | torch.Tensor) -> torch.Tensor:
    """Compute L_degree(softplus(alpha * z)), clamped to [-50, 50].

    softplus maps z from R to (0, inf), placing it in the valid Laguerre domain.
    The outer clamp guards against polynomial growth at large t for high degrees.
    """
    t = F.softplus(z.clamp(-20.0, 20.0) * alpha)
    return _laguerre_poly(degree, t).clamp(-50.0, 50.0)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class LaguerreBlock2D(nn.Module):
    """S2-style shared-projection Laguerre block for 2D images.

    z   = shared_conv(x)                              # one projection shared by all degrees
    out = BN_lin(conv_lin(x))
        + sum_d  gate_d * BN_d( L_d(softplus(a_d * z)) )

    Parameters
    ----------
    in_ch, out_ch   : channel dimensions
    poly_degrees    : list of Laguerre degrees to use, e.g. [2, 3]
    kernel_size     : spatial convolution kernel (3 is fine for MNIST)
    alpha           : initial value of the per-degree learnable sharpness scalar a_d
    """

    def __init__(self, in_ch: int, out_ch: int,
                 poly_degrees: list[int],
                 kernel_size: int = 3,
                 alpha: float = 0.5):
        super().__init__()
        pad = kernel_size // 2

        # Linear path
        self.conv_lin = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn_lin   = nn.BatchNorm2d(out_ch)

        # Shared polynomial projection
        self.shared_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)

        # Per-degree parameters
        self.poly_degrees = poly_degrees
        self.poly_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(alpha)) for _ in poly_degrees
        ])
        self.poly_bns = nn.ModuleList([
            nn.BatchNorm2d(out_ch) for _ in poly_degrees
        ])
        self.poly_gates = nn.ParameterList([
            nn.Parameter(torch.ones(out_ch) * 1e-4) for _ in poly_degrees
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.clamp(-50.0, 50.0)
        out = self.bn_lin(self.conv_lin(x))
        z   = self.shared_conv(x)
        for alpha_d, bn, gate, deg in zip(
            self.poly_alphas, self.poly_bns, self.poly_gates, self.poly_degrees
        ):
            phi = bn(_laguerre_feature(z, deg, alpha_d))
            out = out + gate.view(1, -1, 1, 1) * phi
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TinyLaguerreVNN(nn.Module):
    """Two-block S2-style Laguerre VNN for MNIST.

    Block 1: LaguerreBlock2D(1  -> ch,  degrees) + MaxPool(2) -> [ch,  14, 14]
    Block 2: LaguerreBlock2D(ch -> 2ch, degrees) + MaxPool(2) -> [2ch,  7,  7]
    Global average pool -> FC(2ch -> num_classes)

    With base_ch=8, degrees=[2,3]: ~2.8 K params.
    Two convs per block (lin + shared) vs three for TinyVNN (lin + left + right).
    """

    def __init__(self, num_classes: int = 10, base_ch: int = 8,
                 poly_degrees=None, alpha: float = 0.5):
        super().__init__()
        if poly_degrees is None:
            poly_degrees = [2, 3]
        degrees = list(poly_degrees)
        ch = base_ch

        self.block1     = LaguerreBlock2D(1,  ch,      poly_degrees=degrees, alpha=alpha)
        self.pool1      = nn.MaxPool2d(2)
        self.block2     = LaguerreBlock2D(ch, ch * 2,  poly_degrees=degrees, alpha=alpha)
        self.pool2      = nn.MaxPool2d(2)
        self.classifier = nn.Linear(ch * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = x.mean(dim=(2, 3))          # global average pool
        return self.classifier(x)
