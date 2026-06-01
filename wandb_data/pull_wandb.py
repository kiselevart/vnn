"""
pull_wandb.py — fetch all VNN experiment runs from W&B and save to parquet.

Usage:
    python pull_wandb.py              # fetch runs + history, save parquet
    python pull_wandb.py --no-history # skip per-epoch history (fast)
    python pull_wandb.py --force      # re-download even if cached parquet exists

Output (same directory as this script):
    runs.parquet    — one row per run, summary metrics + config
    history.parquet — one row per (run, epoch), training curves
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb

ENTITY = "kiselevart-mahidol-university"
PROJECTS = ["vnn", "vnn-tools"]
OUT_DIR = Path(__file__).parent

# ── helpers ────────────────────────────────────────────────────────────────

def _safe(d, key, default=None):
    try:
        return d[key]
    except (KeyError, TypeError):
        return default


def _extract_dataset(name: str, config: dict) -> str:
    ds = str(config.get("dataset", "")).lower()
    if ds in ("ucf101", "hmdb51"):
        return ds
    n = name.lower()
    if n.startswith("hmdb51") or "hmdb51" in n:
        return "hmdb51"
    if n.startswith("ucf101") or "ucf101" in n:
        return "ucf101"
    return ds or "unknown"


def _extract_split(name: str, config: dict) -> int:
    v = config.get("split") or config.get("ucf_split")
    if v is not None:
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
    m = re.search(r"split(\d)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 1


def _extract_q(name: str, config: dict) -> Optional[int]:
    v = config.get("Q") or config.get("q")
    if v is not None:
        try:
            return int(v)
        except (ValueError, TypeError):
            pass
    m = re.search(r"[_-]Q(\d+)", name)
    return int(m.group(1)) if m else None


def _extract_fusion(name: str) -> str:
    if "additive" in name.lower():
        return "additive"
    if "fusion" in name.lower():
        return "multiplicative"
    return "unknown"


def _extract_cubic(name: str, config: dict) -> str:
    mode = config.get("cubic_mode", "")
    if config.get("disable_cubic") or config.get("disable") or "no_cubic" in name:
        return "none"
    if mode == "general":
        return "general"
    if mode in ("symmetric", "sym") or "cubic_sym" in name:
        return "sym"
    if "cubic_gen" in name:
        return "general"
    if "cubic_none" in name or "no_cubic" in name:
        return "none"
    return mode or "sym"  # default in vnn_fusion_ho


def _runtime_sec(run) -> Optional[float]:
    try:
        created = pd.Timestamp(run.created_at)
        ended = pd.Timestamp(run.heartbeatAt)
        return (ended - created).total_seconds()
    except Exception:
        return None


# ── parquet sanitizer ──────────────────────────────────────────────────────

# Columns that should be numeric (cast to float, coerce non-numeric to NaN)
_NUMERIC_COLS = {
    "test_acc", "test_acc5", "test_loss", "val_acc", "val_loss",
    "train_acc", "train_loss", "train_grad_norm", "epoch",
    "total_runtime_sec", "weights_max", "weights_mean", "cross_abs_max",
    "config_epochs", "config_lr", "config_batch_size", "config_seed",
    "config_n_lag", "config_n_lag_t", "config_n_lag_s", "config_base_ch",
    "config_total_params",
}


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns; convert remaining object columns to str/None."""
    df = df.copy()
    for col in df.columns:
        if col in _NUMERIC_COLS or col.startswith(("gate_", "bg_")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif df[col].dtype == object:
            # Replace any dict/list/non-scalar with NaN so parquet can write strings
            df[col] = df[col].apply(
                lambda v: v if (v is None or isinstance(v, (str, bool))) else
                          str(v) if not isinstance(v, (dict, list)) else None
            )
    return df


# ── run fetcher ────────────────────────────────────────────────────────────

def fetch_runs(project: str) -> pd.DataFrame:
    api = wandb.Api()
    print(f"  [{project}] fetching run list...", flush=True)
    runs = api.runs(f"{ENTITY}/{project}")

    rows = []
    for run in runs:
        cfg = run.config or {}
        summary = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
        name = run.name or ""

        row: dict = {
            "run_id": run.id,
            "name": name,
            "project": project,
            "state": run.state,
            "created_at": run.created_at,
            # --- derived categorical fields ---
            "dataset": _extract_dataset(name, cfg),
            "split": _extract_split(name, cfg),
            "Q": _extract_q(name, cfg),
            "fusion": _extract_fusion(name),
            "cubic": _extract_cubic(name, cfg),
            # --- key config scalars ---
            "config_model": cfg.get("model"),
            "config_epochs": cfg.get("epochs"),
            "config_lr": cfg.get("lr"),
            "config_batch_size": cfg.get("batch_size"),
            "config_seed": cfg.get("seed"),
            "config_no_amp": cfg.get("no_amp"),
            "config_disable_cubic": cfg.get("disable_cubic"),
            "config_cubic_mode": cfg.get("cubic_mode"),
            "config_n_lag": cfg.get("n_lag"),
            "config_n_lag_t": cfg.get("n_lag_t"),
            "config_n_lag_s": cfg.get("n_lag_s"),
            "config_base_ch": cfg.get("base_ch"),
            "config_run_name": cfg.get("run_name"),
            "config_wandb_group": cfg.get("wandb_group"),
            "config_total_params": cfg.get("total_params"),
            # --- summary metrics ---
            "test_acc": _safe(summary, "test/acc"),
            "test_acc5": _safe(summary, "test/acc5"),
            "test_loss": _safe(summary, "test/loss"),
            "val_acc": _safe(summary, "val/acc"),
            "val_loss": _safe(summary, "val/loss"),
            "train_acc": _safe(summary, "train/acc"),
            "train_loss": _safe(summary, "train/loss"),
            "train_grad_norm": _safe(summary, "train/grad_norm"),
            "epoch": _safe(summary, "epoch"),
            "total_runtime_sec": _runtime_sec(run),
            # --- weight stats ---
            "weights_max": _safe(summary, "weights/max"),
            "weights_mean": _safe(summary, "weights/mean"),
            "cross_abs_max": _safe(summary, "train/cross_abs_max"),
        }

        # gate scalars: gates/b{i}/quad and gates/b{i}/cubic, i=0..8
        for i in range(9):
            row[f"gate_b{i}_quad"] = _safe(summary, f"gates/b{i}/quad")
            row[f"gate_b{i}_cubic"] = _safe(summary, f"gates/b{i}/cubic")

        # block grad norms: train/bg/b{i}/lin and train/bg/b{i}/quad
        for i in range(5):
            row[f"bg_b{i}_lin"] = _safe(summary, f"train/bg/b{i}/lin")
            row[f"bg_b{i}_quad"] = _safe(summary, f"train/bg/b{i}/quad")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  [{project}] {len(df)} runs fetched  (finished={( df.state=='finished').sum()}, "
          f"crashed={(df.state=='crashed').sum()}, running={(df.state=='running').sum()})")
    return df


# ── history fetcher ────────────────────────────────────────────────────────

HISTORY_KEYS = [
    "epoch",
    "train/acc", "train/loss", "train/grad_norm",
    "val/acc",   "val/loss",   "val/grad_norm",
]

def fetch_history(df_runs: pd.DataFrame) -> pd.DataFrame:
    api = wandb.Api()
    finished = df_runs[df_runs["state"] == "finished"].copy()
    print(f"\nFetching per-epoch history for {len(finished)} finished runs...")
    all_hist = []
    for _, row in finished.iterrows():
        path = f"{ENTITY}/{row['project']}/{row['run_id']}"
        try:
            run = api.run(path)
            hist = run.history(keys=HISTORY_KEYS, pandas=True)
            if hist.empty:
                continue
            hist = hist.dropna(how="all").reset_index(drop=True)
            hist["epoch_step"] = hist.index
            hist["run_id"] = row["run_id"]
            hist["name"] = row["name"]
            hist["project"] = row["project"]
            hist["dataset"] = row["dataset"]
            hist["split"] = row["split"]
            hist["Q"] = row["Q"]
            hist["fusion"] = row["fusion"]
            hist["cubic"] = row["cubic"]
            all_hist.append(hist)
            print(f"  {row['name']}: {len(hist)} steps", flush=True)
        except Exception as e:
            print(f"  SKIP {row['name']}: {e}", flush=True)

    if not all_hist:
        return pd.DataFrame()
    return pd.concat(all_hist, ignore_index=True)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pull W&B runs to parquet")
    parser.add_argument("--no-history", action="store_true",
                        help="skip per-epoch history download")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if parquet files already exist")
    args = parser.parse_args()

    runs_path = OUT_DIR / "runs.parquet"
    history_path = OUT_DIR / "history.parquet"

    if runs_path.exists() and not args.force:
        print(f"runs.parquet already exists ({runs_path}). Use --force to re-download.")
        df_runs = pd.read_parquet(runs_path)
        print(f"Loaded {len(df_runs)} runs from cache.")
    else:
        print("Fetching run summaries from W&B...")
        dfs = [fetch_runs(p) for p in PROJECTS]
        df_runs = pd.concat(dfs, ignore_index=True)
        df_runs = _sanitize_for_parquet(df_runs)
        df_runs.to_parquet(runs_path, index=False)
        print(f"\nSaved {len(df_runs)} runs → {runs_path}")

    if not args.no_history:
        if history_path.exists() and not args.force:
            print(f"history.parquet already exists. Use --force to re-download.")
        else:
            df_hist = fetch_history(df_runs)
            if not df_hist.empty:
                df_hist.to_parquet(history_path, index=False)
                print(f"Saved {len(df_hist)} history rows → {history_path}")
            else:
                print("No history data returned.")

    print("\nDone. Load with:")
    print("  import pandas as pd")
    print("  df = pd.read_parquet('wandb_data/runs.parquet')")
    if not args.no_history:
        print("  hist = pd.read_parquet('wandb_data/history.parquet')")


if __name__ == "__main__":
    main()
