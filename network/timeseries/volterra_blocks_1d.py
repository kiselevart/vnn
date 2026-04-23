"""
1D Volterra building blocks for time series classification.

Adapted from network/video_higher_order/volterra_blocks.py — Conv3d/BN3d/Pool3d
replaced with their 1D equivalents. Volterra math primitives are reused as-is:
they operate on the channel dimension and use *shape[2:] unpacking, so they
work correctly on [B, C, T] tensors without modification.

Building blocks:
    VolterraBlock1D    — Single-kernel block (linear + quadratic + optional cubic)
    MultiKernelBlock1D — Multi-kernel block (parallel convs at different temporal scales)
"""

import torch
import torch.nn as nn

from network.video_higher_order.volterra_blocks import (
    volterra_quadratic,
    volterra_cubic_symmetric,
    volterra_cubic_general,
)


def _init_1d_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class VolterraBlock1D(nn.Module):
    """Single-kernel 1D Volterra block for temporal sequences.

    Computes::

        out = [shortcut(x) +] linear(x) + [quad_gate *] quad(x) [+ cubic_gate * cubic(x)]
        out = pool(out)

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        Q: Rank for quadratic decomposition (default 4).
        Qc: Rank for cubic decomposition (default 2).
        stride: If > 1, applies MaxPool1d(2) after summation.
        use_cubic: Enable 3rd-order cubic path.
        cubic_mode: ``'symmetric'`` (a²·b, 2Q channels) or
                    ``'general'`` (a·b·c, 3Q channels).
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Learnable gate on quadratic branch (init 1e-4).
        kernel_size: Temporal convolution kernel size.
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

        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        pad = ks // 2

        self.conv_lin = nn.Conv1d(in_ch, out_ch, ks, padding=pad)
        self.bn_lin = nn.BatchNorm1d(out_ch)

        self.conv_quad = nn.Conv1d(in_ch, 2 * Q * out_ch, ks, padding=pad)
        self.bn_quad = nn.BatchNorm1d(out_ch)
        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        if use_cubic:
            cubic_mult = 3 if cubic_mode == 'general' else 2
            self.conv_cubic = nn.Conv1d(in_ch, cubic_mult * Qc * out_ch, ks, padding=pad)
            self.bn_cubic = nn.BatchNorm1d(out_ch)
            self.cubic_gate = nn.Parameter(torch.ones(out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        self.pool = nn.MaxPool1d(2, 2) if stride > 1 else nn.Identity()
        _init_1d_weights(self)

    def forward(self, x):
        x = x.clamp(-50.0, 50.0)

        out = self.bn_lin(self.conv_lin(x))

        q = self.bn_quad(volterra_quadratic(self.conv_quad(x), self.Q, self.out_ch))
        if self.gate_quadratic:
            q = self.quad_gate.view(1, -1, 1) * q
        out = out + q

        if self.use_cubic:
            fn = volterra_cubic_general if self.cubic_mode == 'general' else volterra_cubic_symmetric
            c = self.bn_cubic(fn(self.conv_cubic(x), self.Qc, self.out_ch))
            out = out + self.cubic_gate.view(1, -1, 1) * c

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)


class MultiKernelBlock1D(nn.Module):
    """Multi-kernel 1D Volterra block with parallel temporal convolutions.

    Each kernel captures a different temporal receptive field; outputs are
    concatenated. Total output channels = ``ch_per_kernel × len(kernels)``.

    Computes::

        out = [shortcut(x) +] concat_linear(x) + [quad_gate *] concat_quad(x)
        out = pool(out)

    Args:
        in_ch: Input channels.
        ch_per_kernel: Output channels per kernel branch.
        kernels: List of kernel sizes, e.g. ``[9, 5, 1]``.
        Q: Rank for quadratic decomposition.
        stride: If > 1, applies MaxPool1d(2) after summation.
        use_shortcut: Add residual shortcut (1×1 conv + BN).
        gate_quadratic: Learnable gate on quadratic branch (init 1e-4).
    """

    def __init__(self, in_ch, ch_per_kernel, kernels, Q=4, stride=1,
                 use_shortcut=False, gate_quadratic=True):
        super().__init__()
        self.Q = Q
        self.ch_per_kernel = ch_per_kernel
        self.out_ch = ch_per_kernel * len(kernels)
        self.use_shortcut = use_shortcut
        self.gate_quadratic = gate_quadratic

        self.lin_convs = nn.ModuleList(
            nn.Conv1d(in_ch, ch_per_kernel, ks, padding=ks // 2)
            for ks in kernels
        )
        self.bn_lin = nn.BatchNorm1d(self.out_ch)

        self.quad_convs = nn.ModuleList(
            nn.Conv1d(in_ch, 2 * Q * ch_per_kernel, ks, padding=ks // 2)
            for ks in kernels
        )
        self.bn_quad = nn.BatchNorm1d(self.out_ch)

        if gate_quadratic:
            self.quad_gate = nn.Parameter(torch.ones(self.out_ch) * 1e-4)

        if use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm1d(self.out_ch),
            )

        self.pool = nn.MaxPool1d(2, 2) if stride > 1 else nn.Identity()
        _init_1d_weights(self)

    def forward(self, x):
        x = x.clamp(-50.0, 50.0)

        lin = self.bn_lin(torch.cat([c(x) for c in self.lin_convs], dim=1))

        quads = [volterra_quadratic(c(x), self.Q, self.ch_per_kernel) for c in self.quad_convs]
        quad = self.bn_quad(torch.cat(quads, dim=1))
        if self.gate_quadratic:
            quad = self.quad_gate.view(1, -1, 1) * quad

        out = lin + quad

        if self.use_shortcut:
            out = self.shortcut(x) + out

        return self.pool(out)
