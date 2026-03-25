"""
video_higher_order — 3D Volterra Neural Network modules for video.

Provides backbone architectures and fusion heads with 2nd + 3rd order
Volterra interactions for video classification.
"""

from .vnn_4block import VNNRgbHO, VNNFusionHO, Backbone4Block, FusionHead
from .vnn_7block import VNNDeep
from .volterra_blocks import (
    VolterraBlock3D, MultiKernelBlock3D, ClassifierHead,
    volterra_quadratic, volterra_cubic_symmetric, volterra_cubic_general, init_vnn_weights,
)

__all__ = [
    # End-to-end models
    "VNNRgbHO", "VNNFusionHO", "VNNDeep",
    # Backbones and heads
    "Backbone4Block", "FusionHead",
    # Building blocks
    "VolterraBlock3D", "MultiKernelBlock3D", "ClassifierHead",
    # Primitives
    "volterra_quadratic", "volterra_cubic_symmetric",
    "volterra_cubic_general", "init_vnn_weights",
]
