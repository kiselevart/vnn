"""
4-block 3D VNN models: backbones, fusion head, and end-to-end classifiers.

Classes:
    Backbone4Block    — 4-block 3D VNN backbone (multi-scale first block)
    FusionHead        — Single-block fusion classifier with differential LR
    SimpleBackbone    — 4-block backbone with optional cubic toggle
    VNNRgbHO          — End-to-end RGB model (Backbone4Block + FusionHead)
    VNNFusionHO       — End-to-end two-stream fusion (RGB + Flow)
    VNNCubicToggle    — End-to-end model with cubic toggle (SimpleBackbone + FusionHead)
"""

import torch
import torch.nn as nn

from .volterra_blocks import MultiKernelBlock3D, VolterraBlock3D, ClassifierHead


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class Backbone4Block(nn.Module):
    """4-block 3D VNN backbone with multi-scale first block.

    Architecture:
        Block 1: Multi-kernel (5×5×5 + 3×3×3 + 1×1×1) → Quadratic → Pool
        Block 2: Single kernel → Quadratic → Pool
        Block 3: Single kernel → Quadratic + Cubic (no pool)
        Block 4: Single kernel → Quadratic + Cubic → Pool

    Outputs feature maps [B, 96, T/8, H/8, W/8].

    Args:
        num_ch: Input channels (3 for RGB, 2 for optical flow).
        cubic_mode: 'symmetric' or 'general' cubic factorization.
    """

    def __init__(self, num_ch=3, cubic_mode='symmetric'):
        super().__init__()

        # Block 1: Multi-kernel, quadratic only
        self.block1 = MultiKernelBlock3D(
            num_ch, ch_per_kernel=8,
            kernels=[(5, 5, 5), (3, 3, 3), (1, 1, 1)],
            Q=4, stride=2,
        )

        # Block 2: Quadratic only
        self.block2 = VolterraBlock3D(24, 32, Q=4, stride=2, use_shortcut=True)

        # Block 3: Quadratic + Cubic (no pool)
        self.block3 = VolterraBlock3D(
            32, 64, Q=4, Qc=2,
            use_cubic=True, cubic_mode=cubic_mode,
            use_shortcut=True,
        )

        # Block 4: Quadratic + Cubic
        self.block4 = VolterraBlock3D(
            64, 96, Q=4, Qc=2, stride=2,
            use_cubic=True, cubic_mode=cubic_mode,
            use_shortcut=True,
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class SimpleBackbone(nn.Module):
    """4-block VNN backbone with optional cubic toggle.

    The ``use_cubic`` flag enables a clean A/B comparison:
        use_cubic=False → Pure 2nd-order (quadratic only)
        use_cubic=True  → 2nd + 3rd order (quadratic + general cubic)

    Args:
        use_cubic: If True, enables general cubic (a·b·c) in all blocks.
    """

    def __init__(self, use_cubic=True):
        super().__init__()

        config = [
            (3,  32, 2),   # Block 1
            (32, 64, 2),   # Block 2
            (64, 64, 1),   # Block 3 (no pool)
            (64, 96, 2),   # Block 4
        ]

        self.layers = nn.ModuleList([
            VolterraBlock3D(
                in_c, out_c, Q=4, Qc=2, stride=s,
                use_cubic=use_cubic, cubic_mode='general',
            )
            for in_c, out_c, s in config
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Fusion head
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """Fusion classification head with Quadratic + Cubic.

    Architecture::

        VolterraBlock3D (Q+C, shortcut, gated) → Pool → Dropout → FC

    Args:
        num_classes: Number of output classes.
        num_ch: Input channels (96 for single-stream, 288 for two-stream fusion).
        cubic_mode: 'symmetric' or 'general' cubic factorization.
    """

    def __init__(self, num_classes, num_ch=3, cubic_mode='symmetric'):
        super().__init__()

        self.block1 = VolterraBlock3D(
            num_ch, 256, Q=2, Qc=2, stride=2,
            use_cubic=True, cubic_mode=cubic_mode,
            use_shortcut=True, gate_quadratic=True,
        )
        self.classifier = ClassifierHead(12544, num_classes)

    def forward(self, x):
        x = self.block1(x)
        return self.classifier(x)

    def get_1x_lr_params(self):
        """Returns all parameters except the final classifier FC layer."""
        skip = {id(p) for p in self.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        """Returns the classifier FC layer parameters (for higher LR)."""
        for p in self.classifier.fc.parameters():
            if p.requires_grad:
                yield p


# ---------------------------------------------------------------------------
# End-to-end models
# ---------------------------------------------------------------------------

class VNNRgbHO(nn.Module):
    """Higher-order RGB video model: Backbone4Block + FusionHead.

    Args:
        num_classes: Number of output classes.
        cubic_mode: 'symmetric' or 'general' cubic factorization.
    """

    def __init__(self, num_classes, cubic_mode='symmetric'):
        super().__init__()
        self.backbone = Backbone4Block(num_ch=3, cubic_mode=cubic_mode)
        self.head = FusionHead(num_classes=num_classes, num_ch=96, cubic_mode=cubic_mode)

    def forward(self, x):
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


class VNNFusionHO(nn.Module):
    """Higher-order two-stream fusion model: RGB + Optical Flow.

    Architecture:
        - RGB stream:  Backbone4Block(3)
        - Flow stream: Backbone4Block(2)
        - Fusion:      cat(rgb, flow, rgb*flow) → FusionHead(num_ch=288)

    Args:
        num_classes: Number of output classes.
        cubic_mode: 'symmetric' or 'general' cubic factorization.
    """

    def __init__(self, num_classes, cubic_mode='symmetric'):
        super().__init__()
        self.model_rgb = Backbone4Block(num_ch=3, cubic_mode=cubic_mode)
        self.model_of = Backbone4Block(num_ch=2, cubic_mode=cubic_mode)
        self.model_fuse = FusionHead(num_classes=num_classes, num_ch=288, cubic_mode=cubic_mode)

    def forward(self, x):
        rgb, flow = x
        out_rgb = self.model_rgb(rgb)
        out_of = self.model_of(flow)
        cross = out_rgb * out_of
        return self.model_fuse(torch.cat((out_rgb, out_of, cross), 1))

    def get_1x_lr_params(self):
        skip = {id(p) for p in self.model_fuse.classifier.fc.parameters()}
        for p in self.parameters():
            if p.requires_grad and id(p) not in skip:
                yield p

    def get_10x_lr_params(self):
        for p in self.model_fuse.classifier.fc.parameters():
            if p.requires_grad:
                yield p


class VNNCubicToggle(nn.Module):
    """SimpleBackbone + FusionHead with optional cubic toggle.

    Args:
        num_classes: Number of output classes.
        use_cubic: If True, enables general cubic in the backbone.
    """

    def __init__(self, num_classes, use_cubic=True):
        super().__init__()
        self.backbone = SimpleBackbone(use_cubic=use_cubic)
        self.head = FusionHead(num_classes=num_classes, num_ch=96)

    def forward(self, x):
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
