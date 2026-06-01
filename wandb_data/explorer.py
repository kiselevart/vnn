"""
General W&B run explorer for VNN experiments.

The explorer works with the cached parquet files produced by pull_wandb.py and
can fetch missing per-epoch histories from W&B on demand.

Example:
    from explorer import WandbExplorer

    ex = WandbExplorer()
    legacy = ex.collect("legacy Q sweep", name_contains="legacy_fusion_Q", dataset="ucf101")
    r3d = ex.collect("r3d", name_contains="r3d", dataset="ucf101")

    ex.summary([legacy, r3d], metric="test_acc")
    ex.plot_history([legacy, r3d], metric="val_acc")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import pandas as pd

ENTITY = "kiselevart-mahidol-university"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent

SUMMARY_ALIASES = {
    "test/acc": "test_acc",
    "test_acc": "test_acc",
    "test/acc5": "test_acc5",
    "test_acc5": "test_acc5",
    "test/loss": "test_loss",
    "test_loss": "test_loss",
    "val/acc": "val_acc",
    "val_acc": "val_acc",
    "val/loss": "val_loss",
    "val_loss": "val_loss",
    "train/acc": "train_acc",
    "train_acc": "train_acc",
    "train/loss": "train_loss",
    "train_loss": "train_loss",
    "train/grad_norm": "train_grad_norm",
    "train_grad_norm": "train_grad_norm",
}

HISTORY_ALIASES = {
    "val_acc": "val/acc",
    "val_loss": "val/loss",
    "train_acc": "train/acc",
    "train_loss": "train/loss",
    "train_grad_norm": "train/grad_norm",
    "val_grad_norm": "val/grad_norm",
}


@dataclass
class RunCollection:
    """Named set of W&B runs selected from runs.parquet."""

    label: str
    runs: pd.DataFrame

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def run_ids(self) -> list[str]:
        return self.runs["run_id"].astype(str).tolist()

    def table(self, columns: Iterable[str] | None = None) -> pd.DataFrame:
        if columns is None:
            columns = [
                "name",
                "project",
                "state",
                "dataset",
                "split",
                "Q",
                "fusion",
                "cubic",
                "test_acc",
                "test_acc5",
                "val_acc",
            ]
        existing = [col for col in columns if col in self.runs.columns]
        return self.runs[existing].copy()


class WandbExplorer:
    """Select, summarize, and plot cached VNN W&B runs."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        entity: str = ENTITY,
        runs_file: str = "runs.parquet",
        history_file: str = "history.parquet",
    ):
        self.data_dir = Path(data_dir)
        self.entity = entity
        self.runs_path = self.data_dir / runs_file
        self.history_path = self.data_dir / history_file
        self.runs = self._load_runs()
        self._history: pd.DataFrame | None = None

    def _load_runs(self) -> pd.DataFrame:
        if not self.runs_path.exists():
            raise FileNotFoundError(
                f"{self.runs_path} not found. Run `python pull_wandb.py --no-history` first."
            )
        runs = pd.read_parquet(self.runs_path)
        if "config_wandb_group" in runs.columns and "group" not in runs.columns:
            runs["group"] = runs["config_wandb_group"]
        if "config_model" in runs.columns and "model" not in runs.columns:
            runs["model"] = runs["config_model"]
        for col in ("name", "project", "state", "dataset", "fusion", "cubic", "group", "model"):
            if col in runs.columns:
                runs[col] = runs[col].fillna("").astype(str)
        return runs

    def reload(self) -> pd.DataFrame:
        self.runs = self._load_runs()
        self._history = None
        return self.runs

    def search(self, text: str, columns: Iterable[str] = ("name", "model", "group")) -> pd.DataFrame:
        pattern = re.escape(text)
        mask = pd.Series(False, index=self.runs.index)
        for col in columns:
            if col in self.runs.columns:
                mask |= self.runs[col].str.contains(pattern, case=False, na=False)
        return self.runs[mask].copy()

    def collect(
        self,
        label: str,
        *,
        name_contains: str | Iterable[str] | None = None,
        name_regex: str | None = None,
        group: str | None = None,
        model: str | None = None,
        dataset: str | None = None,
        project: str | None = None,
        state: str | None = "finished",
        split: int | Iterable[int] | None = None,
        query: str | None = None,
    ) -> RunCollection:
        """Build a named run collection from common filters."""

        runs = self.runs.copy()
        if state is not None and "state" in runs.columns:
            runs = runs[runs["state"].str.lower() == state.lower()]
        if dataset is not None:
            runs = runs[runs["dataset"].str.lower() == dataset.lower()]
        if project is not None:
            runs = runs[runs["project"].str.lower() == project.lower()]
        if group is not None:
            runs = runs[runs["group"].str.contains(group, case=False, na=False, regex=False)]
        if model is not None:
            runs = runs[
                runs["model"].str.contains(model, case=False, na=False, regex=False)
                | runs["name"].str.contains(model, case=False, na=False, regex=False)
            ]
        if name_contains is not None:
            terms = [name_contains] if isinstance(name_contains, str) else list(name_contains)
            mask = pd.Series(True, index=runs.index)
            for term in terms:
                mask &= runs["name"].str.contains(term, case=False, na=False, regex=False)
            runs = runs[mask]
        if name_regex is not None:
            runs = runs[runs["name"].str.contains(name_regex, case=False, na=False, regex=True)]
        if split is not None:
            splits = [split] if isinstance(split, int) else list(split)
            runs = runs[runs["split"].isin(splits)]
        if query is not None:
            runs = runs.query(query)
        runs = runs.sort_values(["dataset", "split", "name"]).reset_index(drop=True)
        return RunCollection(label, runs)

    def summary(
        self,
        collections: RunCollection | Iterable[RunCollection],
        metric: str = "test_acc",
        groupby: Iterable[str] = ("collection", "dataset"),
    ) -> pd.DataFrame:
        data = self._collection_runs(collections)
        metric_col = self._summary_metric(metric)
        grouped = (
            data.groupby(list(groupby), dropna=False)[metric_col]
            .agg(mean="mean", std="std", n="count", min="min", max="max")
            .reset_index()
        )
        return grouped.sort_values(list(groupby)).reset_index(drop=True)

    def plot_summary(
        self,
        collections: RunCollection | Iterable[RunCollection],
        metric: str = "test_acc",
        groupby: str = "collection",
        ax=None,
        figsize: tuple[float, float] = (8, 4),
    ):
        plt = _require_matplotlib()
        data = self._collection_runs(collections)
        metric_col = self._summary_metric(metric)
        stats = data.groupby(groupby, dropna=False)[metric_col].agg(["mean", "std"]).reset_index()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.bar(stats[groupby].astype(str), stats["mean"], yerr=stats["std"].fillna(0), capsize=4)
        ax.set_xlabel(groupby)
        ax.set_ylabel(metric_col)
        ax.set_title(f"{metric_col} by {groupby}")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        return ax

    def history(
        self,
        collection: RunCollection,
        metrics: Iterable[str] = ("val_acc",),
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Return per-step history for a collection, fetching missing runs if needed."""

        metric_cols = [self._history_metric(metric) for metric in metrics]
        needed_cols = ["epoch", "epoch_step", *metric_cols]
        hist = self._load_history()
        present_ids = set(hist["run_id"].astype(str)) if not hist.empty and "run_id" in hist else set()
        missing_metric_cols = [col for col in metric_cols if col not in hist.columns]
        missing = [
            run_id for run_id in collection.run_ids
            if refresh or missing_metric_cols or run_id not in present_ids
        ]
        if missing:
            fetched = self._fetch_history(collection.runs[collection.runs["run_id"].astype(str).isin(missing)], metric_cols)
            if not fetched.empty:
                hist = pd.concat([hist, fetched], ignore_index=True)
                hist = hist.drop_duplicates(subset=["run_id", "epoch_step"], keep="last")
                hist.to_parquet(self.history_path, index=False)
                self._history = hist
        hist = hist[hist["run_id"].astype(str).isin(collection.run_ids)].copy()
        hist["collection"] = collection.label
        keep = [col for col in needed_cols if col in hist.columns]
        meta = ["run_id", "name", "project", "dataset", "split", "Q", "fusion", "cubic", "collection"]
        keep = [col for col in meta if col in hist.columns] + keep
        return hist[keep].sort_values(["name", "epoch_step"]).reset_index(drop=True)

    def plot_history(
        self,
        collections: RunCollection | Iterable[RunCollection],
        metric: str = "val_acc",
        *,
        aggregate: bool = True,
        smooth: int = 1,
        ax=None,
        figsize: tuple[float, float] = (9, 5),
    ):
        """Plot a W&B history metric such as val_acc, val/acc, train_loss, or val_loss."""

        plt = _require_matplotlib()
        cols = [self._history_metric(metric)]
        frames = [self.history(collection, cols) for collection in self._as_list(collections)]
        hist = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        metric_col = cols[0]
        if hist.empty or metric_col not in hist:
            raise ValueError(f"No history data available for metric {metric!r}")
        step_col = "epoch" if "epoch" in hist.columns and hist["epoch"].notna().any() else "epoch_step"
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        if aggregate:
            for label, sub in hist.groupby("collection"):
                curve = sub.groupby(step_col)[metric_col].agg(["mean", "std"]).reset_index()
                y = curve["mean"]
                if smooth > 1:
                    y = y.rolling(smooth, min_periods=1).mean()
                ax.plot(curve[step_col], y, label=label, linewidth=2)
                if curve["std"].notna().any():
                    std = curve["std"].fillna(0)
                    ax.fill_between(curve[step_col], y - std, y + std, alpha=0.12)
        else:
            for (label, name), sub in hist.groupby(["collection", "name"]):
                y = sub[metric_col]
                if smooth > 1:
                    y = y.rolling(smooth, min_periods=1).mean()
                ax.plot(sub[step_col], y, label=f"{label}: {name}", alpha=0.85)
        ax.set_xlabel(step_col)
        ax.set_ylabel(metric_col)
        ax.set_title(metric_col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ax

    def _load_history(self) -> pd.DataFrame:
        if self._history is not None:
            return self._history.copy()
        if self.history_path.exists():
            self._history = pd.read_parquet(self.history_path)
        else:
            self._history = pd.DataFrame()
        return self._history.copy()

    def _fetch_history(self, runs: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
        import wandb

        api = wandb.Api()
        keys = sorted(set(["epoch", *metric_cols]))
        frames = []
        for _, row in runs.iterrows():
            path = f"{self.entity}/{row['project']}/{row['run_id']}"
            try:
                run = api.run(path)
                hist = run.history(keys=keys, pandas=True)
            except Exception as exc:
                print(f"skip {row['name']}: {exc}")
                continue
            if hist.empty:
                continue
            hist = hist.dropna(how="all").reset_index(drop=True)
            hist["epoch_step"] = hist.index
            for col in ["run_id", "name", "project", "dataset", "split", "Q", "fusion", "cubic"]:
                hist[col] = row.get(col)
            frames.append(hist)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _collection_runs(self, collections: RunCollection | Iterable[RunCollection]) -> pd.DataFrame:
        frames = []
        for collection in self._as_list(collections):
            data = collection.runs.copy()
            data["collection"] = collection.label
            frames.append(data)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _as_list(collections: RunCollection | Iterable[RunCollection]) -> list[RunCollection]:
        if isinstance(collections, RunCollection):
            return [collections]
        return list(collections)

    @staticmethod
    def _summary_metric(metric: str) -> str:
        return SUMMARY_ALIASES.get(metric, metric)

    @staticmethod
    def _history_metric(metric: str) -> str:
        return HISTORY_ALIASES.get(metric, metric)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with `pip install matplotlib` "
            "or use `table()` / `summary()` without plotting."
        ) from exc
    return plt
