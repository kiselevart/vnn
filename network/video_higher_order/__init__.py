"""
video_higher_order — 3D Volterra Neural Network modules for video.

Provides backbone architectures and fusion heads with 2nd + 3rd order
Volterra interactions for video classification.
"""

from .vnn_4block import VNNRgbHO, VNNFusionHO, Backbone4Block, FusionHead
from .volterra_blocks import (
    VolterraBlock3D, MultiKernelBlock3D, ClassifierHead,
    volterra_quadratic, volterra_cubic_symmetric, volterra_cubic_general, init_vnn_weights,
)
from .lvn_blocks import (
    LVNRgb, LVNFusion,
    lvn_rgb_signed,
    lvn_fusion_signed,
)
from .laguerre_conv import (
    LaguerreConv3d,
    LaguerreVolterraBlock3D,
    lvn_laguerre_rgb, lvn_laguerre_fusion,
    lvn_monomial_rgb, lvn_monomial_fusion,
    LaguerreConv3d_Full,
    lvn_laguerre_full_rgb, lvn_laguerre_full_fusion,
)

__all__ = [
    # Volterra end-to-end models
    "VNNRgbHO", "VNNFusionHO",
    # Backbones and heads
    "Backbone4Block", "FusionHead",
    # Volterra building blocks
    "VolterraBlock3D", "MultiKernelBlock3D", "ClassifierHead",
    "volterra_quadratic", "volterra_cubic_symmetric",
    "volterra_cubic_general", "init_vnn_weights",
    # LVN models (Volterra interactions)
    "LVNRgb", "LVNFusion",
    "lvn_rgb_signed", "lvn_fusion_signed",
    # Proper Laguerre-basis models (temporal only)
    "LaguerreConv3d", "LaguerreVolterraBlock3D",
    "lvn_laguerre_rgb", "lvn_laguerre_fusion",
    "lvn_monomial_rgb", "lvn_monomial_fusion",
    # Full 3D Laguerre basis (Tucker decomposition over T, H, W)
    "LaguerreConv3d_Full",
    "lvn_laguerre_full_rgb", "lvn_laguerre_full_fusion",
]
