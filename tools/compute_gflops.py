"""
Compute GFLOPs for every model variant used in ablations.
Run from vnn/ directory: python compute_gflops.py
Requires fvcore: pip install fvcore
"""
import sys, torch
import torch.nn as nn

sys.path.insert(0, ".")

from network.video_higher_order.vnn_4block import VNNFusionHO, VNNAdditiveFusionHO, VNNSmallAdditiveFusion
from network.video_higher_order.vnn_legacy import VNNLegacyFusion
from network.video.established_models import R2Plus1DNet, R3DNet, ResNet50FrameAvg, SmallR3D, SmallR2Plus1D
from network.video.i3d import I3DTwoStream
from network.video_higher_order import (
    lvn_laguerre_fusion,
    lvn_laguerre_full_fusion,
    lvn_legendre_fusion,
    lvn_chebyshev_fusion,
    lvn_hermite_fusion,
)

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("WARNING: fvcore not found, falling back to manual MAC hooks (less accurate)")


NUM_CLASSES = 101
T, H, W = 16, 112, 112
DEVICE = torch.device("cpu")


def gflops_fusion(model):
    model.eval().to(DEVICE)
    rgb  = torch.zeros(1, 3, T, H, W)
    flow = torch.zeros(1, 2, T, H, W)
    if HAS_FVCORE:
        fa = FlopCountAnalysis(model, ([rgb, flow],))
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    return _manual_hooks(model, [rgb, flow])


def gflops_single(model):
    model.eval().to(DEVICE)
    x = torch.zeros(1, 3, T, H, W)
    if HAS_FVCORE:
        fa = FlopCountAnalysis(model, (x,))
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    return _manual_hooks(model, x)


def _manual_hooks(model, inp):
    total_macs = 0

    def conv_hook(m, _i, out):
        nonlocal total_macs
        k = m.kernel_size if hasattr(m.kernel_size, "__len__") else (m.kernel_size,)
        ops = m.in_channels // m.groups
        for ki in k:
            ops *= ki
        total_macs += ops * out[0].numel()

    def linear_hook(m, _i, out):
        nonlocal total_macs
        total_macs += m.in_features * out[0].numel()

    hooks = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            model(inp)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()
    return 2 * total_macs / 1e9


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


rows = []  # (group, label, gflops, params_M)

# ── Q sweep (vnn_legacy_fusion) ──────────────────────────────────────────────
for Q in [1, 2, 4, 8, 16]:
    m = VNNLegacyFusion(num_classes=NUM_CLASSES, Q=Q)
    g = gflops_fusion(m)
    rows.append(("Q sweep", f"vnn_legacy_fusion  Q={Q}", g, count_params(m)))

# ── Fusion ablation — cubic ON ────────────────────────────────────────────────
for label, cls in [("multiplicative (cubic ON)", VNNFusionHO),
                   ("additive      (cubic ON)", VNNAdditiveFusionHO)]:
    m = cls(num_classes=NUM_CLASSES, cubic_mode="symmetric", use_cubic=True)
    g = gflops_fusion(m)
    rows.append(("Fusion ablation", f"vnn_{label}", g, count_params(m)))

# ── Fusion ablation — no cubic ────────────────────────────────────────────────
for label, cls in [("multiplicative (no cubic)", VNNFusionHO),
                   ("additive      (no cubic)", VNNAdditiveFusionHO)]:
    m = cls(num_classes=NUM_CLASSES, cubic_mode="symmetric", use_cubic=False)
    g = gflops_fusion(m)
    rows.append(("Fusion ablation", f"vnn_{label}", g, count_params(m)))

# ── Cubic ablation (vnn_fusion_ho multiplicative only) ───────────────────────
for label, use_cubic, cubic_mode in [
    ("none (Q=6)",       False, "symmetric"),
    ("symmetric (a²b)",  True,  "symmetric"),
    ("general (abc)",    True,  "general"),
]:
    m = VNNFusionHO(num_classes=NUM_CLASSES, cubic_mode=cubic_mode, use_cubic=use_cubic)
    g = gflops_fusion(m)
    rows.append(("Cubic ablation", f"vnn_fusion_ho  cubic={label}", g, count_params(m)))

# ── Baselines ────────────────────────────────────────────────────────────────
for label, cls, fn in [
    ("R(2+1)D-18",       R2Plus1DNet,      gflops_single),
    ("R3D-18",           R3DNet,           gflops_single),
    ("ResNet50FrameAvg", ResNet50FrameAvg, gflops_single),
    ("SmallR3D",         SmallR3D,         gflops_single),
    ("SmallR2Plus1D",    SmallR2Plus1D,    gflops_single),
]:
    m = cls(num_classes=NUM_CLASSES)
    g = fn(m)
    rows.append(("Baselines", label, g, count_params(m)))

m = I3DTwoStream(num_classes=NUM_CLASSES, clip_len=T)
rows.append(("Baselines", "I3D Two-Stream (RGB+Flow)", gflops_fusion(m), count_params(m)))

# ── Small VNN variants ───────────────────────────────────────────────────────
for label, build in [
    ("vnn_small_legacy_fusion  (Q=1, Q_fuse=1)",  lambda: VNNLegacyFusion(NUM_CLASSES, Q=1, Q_fusion=1)),
    ("vnn_small_additive_fusion (Q=1, ch/k=4)",   lambda: VNNSmallAdditiveFusion(NUM_CLASSES)),
]:
    m = build()
    rows.append(("Small VNN", label, gflops_fusion(m), count_params(m)))

# ── LVN / Ortho basis models ─────────────────────────────────────────────────
for label, build in [
    ("TLVN  (Laguerre temporal, n_lag=4)",          lambda: lvn_laguerre_fusion(NUM_CLASSES, clip_len=T, n_lag=4)),
    ("LVN   (Laguerre full T+H+W, n_t=4 n_s=2)",   lambda: lvn_laguerre_full_fusion(NUM_CLASSES, clip_len=T, n_lag_t=4, n_lag_s=2)),
    ("Legendre full (n_t=4 n_s=2)",                 lambda: lvn_legendre_fusion(NUM_CLASSES, clip_len=T, n_poly=4, n_poly_s=2)),
    ("Chebyshev full (n_t=4 n_s=2)",                lambda: lvn_chebyshev_fusion(NUM_CLASSES, clip_len=T, n_poly=4, n_poly_s=2)),
    ("Hermite   full (n_t=4 n_s=2)",                lambda: lvn_hermite_fusion(NUM_CLASSES, clip_len=T, n_poly=4, n_poly_s=2)),
]:
    m = build()
    rows.append(("LVN/Ortho", label, gflops_fusion(m), count_params(m)))

# ── Print table ──────────────────────────────────────────────────────────────
print()
print(f"{'Group':<20} {'Model':<45} {'GFLOPs/view':>12} {'Params (M)':>10}")
print("-" * 92)
last_group = None
for group, label, g, p in rows:
    if group != last_group:
        if last_group is not None:
            print()
        last_group = group
    gstr = f"{g:.2f}" if g is not None else "N/A"
    print(f"{group:<20} {label:<45} {gstr:>12} {p:>10.2f}")
print()
