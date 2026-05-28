#!/bin/bash
# Measure peak VRAM per GPU for each LVN/TLVN/ortho fusion model.
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

MODELS = [
    ("TLVN",      lambda: lvn_laguerre_fusion(num_classes=101, clip_len=16, n_lag=4)),
    ("LVN",       lambda: lvn_laguerre_full_fusion(num_classes=101, clip_len=16, n_lag_t=4, n_lag_s=2)),
    ("Legendre",  lambda: lvn_legendre_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
    ("Chebyshev", lambda: lvn_chebyshev_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
    ("Hermite",   lambda: lvn_hermite_fusion(num_classes=101, clip_len=16, n_poly=4, n_poly_s=2)),
]

device = torch.device("cuda:0")
B, T, H, W = 8, 16, 112, 112

print(f"{'Model':<12}  {'Peak VRAM':>12}  {'Params':>10}")
print("-" * 40)
for name, build in MODELS:
    torch.cuda.reset_peak_memory_stats(device)
    net = build().to(device).train()
    rgb  = torch.randn(B, 3, T, H, W, device=device)
    flow = torch.randn(B, 2, T, H, W, device=device)
    out  = net((rgb, flow))
    F.cross_entropy(out, torch.zeros(B, dtype=torch.long, device=device)).backward()
    peak   = torch.cuda.max_memory_reserved(device) / 1024**2
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"{name:<12}  {peak:>9,.0f} MB  {params:>8.1f} M")
    del net, rgb, flow, out
    torch.cuda.empty_cache()
PYEOF
