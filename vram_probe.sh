#!/bin/bash
# Measure peak VRAM per GPU for all ablation models (BS=8, float32, forward+backward).
# In 4-GPU DDP each GPU sees BS=2; this script uses BS=8 on one GPU to be conservative.
# Usage:
#   cd vnn
#   bash vram_probe.sh [GPU_ID]   # default GPU 0

GPU="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python3 - <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
import torch.nn.functional as F
from network.video_higher_order import (
    lvn_laguerre_fusion,
    lvn_laguerre_full_fusion,
    lvn_legendre_fusion,
    lvn_chebyshev_fusion,
    lvn_hermite_fusion,
)
from network.video_higher_order.vnn_4block import VNNFusionHO, VNNAdditiveFusionHO
from network.video.established_models import R3DNet, R2Plus1DNet, ResNet50FrameAvg, SmallR3D, SmallR2Plus1D
from network.video.i3d import I3DTwoStream

TWO_STREAM = True
ONE_STREAM = False

MODELS = [
    # --- Diving48 ablation models ---
    ("VNN/fusion",    TWO_STREAM, lambda: VNNFusionHO(num_classes=48, use_cubic=False)),
    ("VNN/additive",  TWO_STREAM, lambda: VNNAdditiveFusionHO(num_classes=48, use_cubic=False)),
    ("SmallR3D-48",   ONE_STREAM, lambda: SmallR3D(num_classes=48)),
    ("R3D-18-48",     ONE_STREAM, lambda: R3DNet(num_classes=48)),
    # --- LVN/ortho models (UCF101/HMDB51) ---
    ("TLVN",          TWO_STREAM, lambda: lvn_laguerre_fusion(num_classes=101, clip_len=16, n_lag=4)),
    ("LVN",           TWO_STREAM, lambda: lvn_laguerre_full_fusion(num_classes=101, clip_len=16, n_lag_t=4, n_lag_s=2)),
    ("Legendre",      TWO_STREAM, lambda: lvn_legendre_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
    ("Chebyshev",     TWO_STREAM, lambda: lvn_chebyshev_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
    ("Hermite",       TWO_STREAM, lambda: lvn_hermite_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
    ("SmallR3D",      ONE_STREAM, lambda: SmallR3D(num_classes=101)),
    ("SmallR2+1D",    ONE_STREAM, lambda: SmallR2Plus1D(num_classes=101)),
    # --- Large baselines (missing_baselines.sh: UCF101/HMDB51 multi-split) ---
    ("R3D-18",        ONE_STREAM, lambda: R3DNet(num_classes=101)),
    ("R(2+1)D-18",    ONE_STREAM, lambda: R2Plus1DNet(num_classes=101)),
    ("ResNet50-avg",  ONE_STREAM, lambda: ResNet50FrameAvg(num_classes=101)),
    ("I3D-2stream",   TWO_STREAM, lambda: I3DTwoStream(num_classes=101)),
]

device = torch.device("cuda:0")
torch.cuda.set_device(device)
B, T, H, W = 8, 16, 112, 112

print(f"{'Model':<14}  {'Peak VRAM (BS=8)':>16}  {'Per-GPU DDP (BS=2)':>18}  {'Params':>10}")
print("-" * 68)
for name, two_stream, build in MODELS:
    torch.cuda.reset_peak_memory_stats(device)

    net = build().to(device).train()
    rgb = torch.randn(B, 3, T, H, W, device=device)

    if two_stream:
        flow = torch.randn(B, 2, T, H, W, device=device)
        out = net((rgb, flow))
    else:
        out = net(rgb)

    labels = torch.zeros(B, dtype=torch.long, device=device)
    if isinstance(out, (tuple, list)) and len(out) == 3 and isinstance(out[2], list):
        # I3DTwoStream training output: (rgb_main, flow_main, aux_list)
        rgb_main, flow_main, aux_list = out
        loss = F.cross_entropy(rgb_main, labels) + F.cross_entropy(flow_main, labels)
        for aux in aux_list:
            loss = loss + 0.3 * F.cross_entropy(aux, labels)
    else:
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits, labels)
    loss.backward()

    peak_mb = torch.cuda.max_memory_reserved(device) / 1024**2
    params  = sum(p.numel() for p in net.parameters()) / 1e6
    # DDP splits batch across GPUs; activations scale roughly linearly with BS
    ddp_est = peak_mb / 4
    print(f"{name:<14}  {peak_mb:>13,.0f} MB  {ddp_est:>15,.0f} MB  {params:>8.1f} M")

    del net, rgb, out
    if two_stream:
        del flow
    torch.cuda.empty_cache()
PYEOF