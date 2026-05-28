#!/bin/bash
# Measure peak VRAM per GPU for each LVN/TLVN/ortho fusion model.
# Runs a synthetic forward+backward (BS=8, same as DDP per-GPU batch).
#
# Usage:
#   cd vnn
#   bash vram_probe.sh [GPU_ID]   # default GPU 0

GPU="${1:-0}"

CUDA_VISIBLE_DEVICES="$GPU" python3 - <<'PYEOF'
import torch, sys
sys.path.insert(0, ".")
from types import SimpleNamespace
from utils.model_factory import get_model

MODELS = [
    ("TLVN",      dict(model="lvn_laguerre_fusion",     n_lag=4,   laguerre_lr_mult=2.0)),
    ("LVN",       dict(model="lvn_laguerre_full_fusion", n_lag_t=4, n_lag_s=2)),
    ("Legendre",  dict(model="lvn_legendre_fusion",      n_lag=4,   n_lag_s=2)),
    ("Chebyshev", dict(model="lvn_chebyshev_fusion",     n_lag=4,   n_lag_s=2)),
    ("Hermite",   dict(model="lvn_hermite_fusion",       n_lag=4,   n_lag_s=2)),
]

device = torch.device("cuda:0")
B, T, H, W = 8, 16, 112, 112

print(f"{'Model':<12}  {'Peak VRAM':>12}  {'Params':>10}")
print("-" * 40)
for name, kw in MODELS:
    torch.cuda.reset_peak_memory_stats(0)
    args = SimpleNamespace(
        task="video", num_classes=101, Q=4, clip_len=T,
        cubic_mode="symmetric", disable_cubic=False,
        alpha=1.0, n_lag=None, n_lag_t=None, n_lag_s=None,
        laguerre_lr_mult=3.0, **kw
    )
    net = get_model(args, device).train()
    rgb  = torch.randn(B, 3, T, H, W, device=device)
    flow = torch.randn(B, 2, T, H, W, device=device)
    out  = net((rgb, flow))
    torch.nn.functional.cross_entropy(
        out, torch.zeros(B, dtype=torch.long, device=device)
    ).backward()
    peak   = torch.cuda.max_memory_reserved(0) / 1024**2
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"{name:<12}  {peak:>9,.0f} MB  {params:>8.1f} M")
    del net, rgb, flow, out
    torch.cuda.empty_cache()
PYEOF
