#!/usr/bin/env python3
"""
Run VNN1D on a named suite of UCR/UEA benchmarks and print a results table.

Each dataset runs as an isolated subprocess so W&B sessions, process state,
and GPU memory are all cleanly separated between runs.

Suites
------
quick     ECG5000 + FordA                        (2 datasets, ~5 min)
standard  5 datasets from easy to hard           (default)
full      7 datasets                             (~60–120 min depending on GPU)

Usage
-----
python benchmark.py                              # standard suite
python benchmark.py --suite quick
python benchmark.py --suite full
python benchmark.py --datasets ECG5000 NATOPS   # ad-hoc list
python benchmark.py --epochs 200 --suite standard  # epoch override
python benchmark.py --download-only             # fetch datasets, no training
python benchmark.py --no-wandb                  # suppress W&B (offline mode)
"""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

@dataclass
class DSConfig:
    name: str
    base_ch: int
    epochs: int
    lr: float       = 1e-3
    difficulty: str = ""
    in_ch: int      = 0     # 0 = auto-detected from file
    seq_len: int    = 0
    n_classes: int  = 0
    n_train: int    = 0


# Each entry tuned so the model is not over- or under-parameterised for the dataset.
_STANDARD: list[DSConfig] = [
    DSConfig("ECG5000",                  base_ch=4, epochs=150,
             difficulty="easy",        in_ch=1, seq_len=140,  n_classes=5,  n_train=500),
    DSConfig("FordA",                    base_ch=4, epochs=150,
             difficulty="medium",      in_ch=1, seq_len=500,  n_classes=2,  n_train=3601),
    DSConfig("ArticularyWordRecognition",base_ch=8, epochs=200,
             difficulty="medium",      in_ch=9, seq_len=144,  n_classes=25, n_train=275),
    DSConfig("NATOPS",                   base_ch=8, epochs=200,
             difficulty="medium",      in_ch=24,seq_len=51,   n_classes=6,  n_train=180),
    DSConfig("EthanolConcentration",     base_ch=8, epochs=300,
             difficulty="hard",        in_ch=3, seq_len=1751, n_classes=4,  n_train=261),
]

_FULL: list[DSConfig] = _STANDARD + [
    DSConfig("UWaveGestureLibrary",      base_ch=8, epochs=200,
             difficulty="medium-hard", in_ch=3, seq_len=315,  n_classes=8,  n_train=120),
    DSConfig("ElectricDevices",          base_ch=4, epochs=150,
             difficulty="medium",      in_ch=1, seq_len=96,   n_classes=7,  n_train=8926),
]

SUITES: dict[str, list[DSConfig]] = {
    "quick":    _STANDARD[:2],
    "standard": _STANDARD,
    "full":     _FULL,
}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def ensure_datasets(configs: list[DSConfig], root: str) -> list[DSConfig]:
    """Download any missing datasets; return configs that are ready."""
    missing = [c for c in configs
               if not os.path.exists(os.path.join(root, c.name, f"{c.name}_TRAIN.ts"))]
    if missing:
        names = [c.name for c in missing]
        print(f"Downloading {len(missing)} missing dataset(s): {', '.join(names)}")
        cmd = [sys.executable, "tools/download_ts_datasets.py",
               "--root", root, "--dataset"] + names
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print("WARNING: some downloads failed — skipping those datasets.")
            configs = [c for c in configs
                       if os.path.exists(os.path.join(root, c.name, f"{c.name}_TRAIN.ts"))]
    return configs


# ---------------------------------------------------------------------------
# Training runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    dataset: str
    difficulty: str
    in_ch: int
    seq_len: int
    n_classes: int
    n_train: int
    params: Optional[int]     = None
    best_val_acc: Optional[float] = None
    test_acc: Optional[float] = None
    elapsed_min: Optional[float] = None
    error: Optional[str]      = None


_PARAMS_RE    = re.compile(r"Total Parameters:\s*([\d,]+)")
_TEST_ACC_RE  = re.compile(r"Test Result.*Acc:\s*([\d.]+)%")
_VAL_ACC_RE   = re.compile(r"V_Acc:\s*([\d.]+)%")


def run_dataset(cfg: DSConfig, args: argparse.Namespace,
                group: str, run_idx: int) -> Result:
    result = Result(
        dataset    = cfg.name,
        difficulty = cfg.difficulty,
        in_ch      = cfg.in_ch,
        seq_len    = cfg.seq_len,
        n_classes  = cfg.n_classes,
        n_train    = cfg.n_train,
    )

    epochs  = args.epochs  if args.epochs  else cfg.epochs
    base_ch = args.base_ch if args.base_ch else cfg.base_ch
    lr      = args.lr      if args.lr      else cfg.lr

    run_name = f"{cfg.name}_{group}"

    cmd = [
        sys.executable, "train.py",
        "--dataset",    cfg.name,
        "--model",      args.model,
        "--run_name",   run_name,
        "--epochs",     str(epochs),
        "--base_ch",    str(base_ch),
        "--lr",         str(lr),
        "--batch_size", str(args.batch_size),
        "--num_workers",str(args.num_workers),
        "--wandb_group",group,
        "--Q",          str(args.Q),
        "--Qc",         str(args.Qc),
        "--alpha",      str(args.alpha),
    ]
    if args.disable_cubic:
        cmd.append("--disable_cubic")
    if args.cubic_mode != "symmetric":
        cmd += ["--cubic_mode", args.cubic_mode]
    if args.poly_degrees:
        cmd += ["--poly_degrees"] + [str(d) for d in args.poly_degrees]

    env = os.environ.copy()
    if args.no_wandb:
        env["WANDB_MODE"] = "disabled"
    if args.ucr_root:
        env["UCR_ROOT"] = args.ucr_root

    print(f"\n{'='*64}")
    print(f"[{run_idx}] {cfg.name}  ({cfg.difficulty})  "
          f"C={cfg.in_ch} T={cfg.seq_len} cls={cfg.n_classes} train={cfg.n_train}")
    print(f"    model={args.model}  base_ch={base_ch}  Q={args.Q}  Qc={args.Qc}  epochs={epochs}  lr={lr}")
    print(f"{'='*64}")

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=False, text=True, env=env)
    result.elapsed_min = (time.time() - t0) / 60

    # Re-run capturing output just to parse numbers (already printed live above
    # because capture_output=False; re-run is fast since it will hit the
    # already-existing checkpoint... actually we can't do that easily).
    # Instead, run with Popen and tee output.

    if proc.returncode != 0:
        result.error = f"exit code {proc.returncode}"
    return result


def run_dataset_tee(cfg: DSConfig, args: argparse.Namespace,
                    group: str, run_idx: int) -> Result:
    """Run training, stream output to terminal, and capture it for parsing."""
    result = Result(
        dataset    = cfg.name,
        difficulty = cfg.difficulty,
        in_ch      = cfg.in_ch,
        seq_len    = cfg.seq_len,
        n_classes  = cfg.n_classes,
        n_train    = cfg.n_train,
    )

    epochs  = args.epochs  if args.epochs  else cfg.epochs
    base_ch = args.base_ch if args.base_ch else cfg.base_ch
    lr      = args.lr      if args.lr      else cfg.lr

    run_name = f"{cfg.name}_{group}"

    cmd = [
        sys.executable, "train.py",
        "--dataset",    cfg.name,
        "--model",      args.model,
        "--run_name",   run_name,
        "--epochs",     str(epochs),
        "--base_ch",    str(base_ch),
        "--lr",         str(lr),
        "--batch_size", str(args.batch_size),
        "--num_workers",str(args.num_workers),
        "--wandb_group",group,
        "--Q",          str(args.Q),
        "--Qc",         str(args.Qc),
        "--alpha",      str(args.alpha),
    ]
    if args.disable_cubic:
        cmd.append("--disable_cubic")
    if args.cubic_mode != "symmetric":
        cmd += ["--cubic_mode", args.cubic_mode]
    if args.poly_degrees:
        cmd += ["--poly_degrees"] + [str(d) for d in args.poly_degrees]

    env = os.environ.copy()
    if args.no_wandb:
        env["WANDB_MODE"] = "disabled"
    if args.ucr_root:
        env["UCR_ROOT"] = args.ucr_root

    print(f"\n{'='*64}")
    print(f"[{run_idx}] {cfg.name}  ({cfg.difficulty})  "
          f"C={cfg.in_ch} T={cfg.seq_len} cls={cfg.n_classes} train={cfg.n_train}")
    print(f"    model={args.model}  base_ch={base_ch}  Q={args.Q}  Qc={args.Qc}  epochs={epochs}  lr={lr}")
    print(f"{'='*64}", flush=True)

    t0 = time.time()
    captured = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, env=env, bufsize=1) as proc:
        for line in proc.stdout:  # type: ignore[union-attr]
            print(line, end="", flush=True)
            captured.append(line)
    result.elapsed_min = (time.time() - t0) / 60

    if proc.returncode != 0:
        result.error = f"exit code {proc.returncode}"
        return result

    log = "".join(captured)

    m = _PARAMS_RE.search(log)
    if m:
        result.params = int(m.group(1).replace(",", ""))

    m = _TEST_ACC_RE.search(log)
    if m:
        result.test_acc = float(m.group(1))

    # best val acc = max across all epoch lines
    val_accs = [float(x) for x in _VAL_ACC_RE.findall(log)]
    if val_accs:
        result.best_val_acc = max(val_accs)

    return result


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table(results: list[Result]) -> None:
    print(f"\n\n{'='*90}")
    print("BENCHMARK RESULTS")
    print(f"{'='*90}")
    header = (f"{'Dataset':<32} {'Diff':<12} {'C':>2} {'T':>5} {'Cls':>3} "
              f"{'Params':>8} {'Val%':>6} {'Test%':>6} {'Min':>5}")
    print(header)
    print("-" * 90)
    for r in results:
        if r.error:
            print(f"{r.dataset:<32} {'ERROR: ' + r.error}")
            continue
        params  = f"{r.params:,}"    if r.params      is not None else "—"
        val_acc = f"{r.best_val_acc:.1f}" if r.best_val_acc is not None else "—"
        test_acc= f"{r.test_acc:.1f}"     if r.test_acc    is not None else "—"
        elapsed = f"{r.elapsed_min:.1f}"  if r.elapsed_min is not None else "—"
        print(f"{r.dataset:<32} {r.difficulty:<12} {r.in_ch:>2} {r.seq_len:>5} "
              f"{r.n_classes:>3} {params:>8} {val_acc:>6} {test_acc:>6} {elapsed:>5}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=list(SUITES.keys()), default="standard")
    p.add_argument("--datasets", nargs="+", metavar="NAME",
                   help="Ad-hoc list of dataset names (overrides --suite).")
    p.add_argument("--epochs",    type=int,   default=0,
                   help="Override epoch count for all datasets (0 = use suite defaults).")
    p.add_argument("--base_ch",   type=int,   default=0,
                   help="Override base_ch for all datasets (0 = use suite defaults).")
    p.add_argument("--lr",        type=float, default=0.0,
                   help="Override lr for all datasets (0 = use suite defaults).")
    p.add_argument("--model",     type=str,   default="vnn_1d",
                   help="Model name passed to train.py (e.g. vnn_1d, laguerre_vnn_1d, fcn, resnet1d, inceptiontime).")
    p.add_argument("--Q",         type=int,   default=2)
    p.add_argument("--Qc",        type=int,   default=1)
    p.add_argument("--poly_degrees", type=int, nargs="+", default=None,
                   help="Laguerre degrees for laguerre_vnn_1d, e.g. --poly_degrees 2 3.")
    p.add_argument("--alpha",     type=float, default=1.0,
                   help="Softplus scale for laguerre_vnn_1d.")
    p.add_argument("--batch_size",type=int,   default=32)
    p.add_argument("--num_workers",type=int,  default=4)
    p.add_argument("--disable_cubic", action="store_true")
    p.add_argument("--cubic_mode", type=str, default="symmetric",
                   choices=["symmetric", "general"],
                   help="Cubic interaction mode for vnn_1d.")
    p.add_argument("--ucr_root",  type=str,
                   default=os.environ.get("UCR_ROOT", "./data/ucr"))
    p.add_argument("--download-only", action="store_true",
                   help="Download missing datasets and exit.")
    p.add_argument("--wandb_group", type=str, default=None,
                   help="W&B group name for all runs in this benchmark invocation.")
    p.add_argument("--no-wandb",  action="store_true", dest="no_wandb",
                   help="Run W&B in offline mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build config list
    if args.datasets:
        # Ad-hoc: look up in known configs, fall back to defaults
        known = {c.name: c for suite in SUITES.values() for c in suite}
        configs = []
        for name in args.datasets:
            if name in known:
                configs.append(known[name])
            else:
                # Unknown dataset — minimal config, let suite defaults handle it
                configs.append(DSConfig(name, base_ch=8, epochs=150))
    else:
        configs = SUITES[args.suite]

    # Download missing datasets
    configs = ensure_datasets(configs, args.ucr_root)
    if not configs:
        print("No datasets available. Exiting.")
        sys.exit(1)

    if args.download_only:
        print("Downloads complete.")
        return

    # W&B group: prefer explicit --wandb_group, fall back to auto-generated name
    group = args.wandb_group or f"bench_{datetime.now().strftime('%m%d-%H%M')}"
    print(f"\nBenchmark group: {group}  ({len(configs)} datasets)")

    results: list[Result] = []
    total_t0 = time.time()

    for i, cfg in enumerate(configs, 1):
        result = run_dataset_tee(cfg, args, group, i)
        results.append(result)
        # Print running partial table after each dataset
        print_table(results)

    total_min = (time.time() - total_t0) / 60
    print(f"\nTotal benchmark time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
