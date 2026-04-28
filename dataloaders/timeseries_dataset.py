"""
Dataset and loader utilities for UCR/UEA time series classification benchmarks.

Data format expected on disk:
    <root>/<DatasetName>/<DatasetName>_TRAIN.ts
    <root>/<DatasetName>/<DatasetName>_TEST.ts

Loaded via the `aeon` library (pip install aeon).
The aeon library returns X as numpy [N, C, T] float64 arrays, which are
converted to [N, C, T] float32 tensors with 0-indexed integer labels.

Normalization: per-sample z-score along the time axis (standard for deep
learning on UCR/UEA benchmarks; keeps the model invariant to amplitude scale).

Augmentation: Gaussian jitter only — simple and effective for 1D series.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fill_nan(X: np.ndarray) -> None:
    """In-place forward-fill NaN along time axis, then zero-fill any remainder."""
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            s = X[i, c]
            nan_mask = np.isnan(s)
            if not nan_mask.any():
                continue
            # Forward fill: propagate last valid observation
            idx = np.where(~nan_mask, np.arange(len(s)), 0)
            np.maximum.accumulate(idx, out=idx)
            s[nan_mask] = s[idx[nan_mask]]
            # Zero-fill any leading NaNs (no valid predecessor)
            s[np.isnan(s)] = 0.0


def _pad_ragged(X_list: list, max_len: int) -> np.ndarray:
    """Pad a list of [C, T_i] arrays to a [N, C, max_len] float32 array."""
    C = X_list[0].shape[0]
    out = np.zeros((len(X_list), C, max_len), dtype=np.float32)
    for i, x in enumerate(X_list):
        out[i, :, :x.shape[-1]] = x
    return out


def _zscore(X: np.ndarray) -> np.ndarray:
    """Per-sample z-score along the time axis. Constant series → unchanged."""
    mean = X.mean(axis=-1, keepdims=True)
    std  = X.std(axis=-1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)   # avoid division by zero
    return (X - mean) / std


def _encode_labels(y_train, y_test):
    """Map string or arbitrary labels to contiguous 0-indexed integers."""
    classes  = sorted(set(y_train) | set(y_test), key=lambda v: str(v))
    label_map = {c: i for i, c in enumerate(classes)}
    return (
        np.array([label_map[c] for c in y_train], dtype=np.int64),
        np.array([label_map[c] for c in y_test],  dtype=np.int64),
        classes,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TSDataset(Dataset):
    """PyTorch Dataset for time series classification.

    Args:
        X: Float array [N, C, T].
        y: Int array [N] with 0-indexed class labels.
        augment: Enable jitter augmentation (Gaussian noise on each sample).
        normalize: Apply per-sample z-score normalization.
        jitter_sigma: Noise std for jitter augmentation (default 0.03).
    """

    def __init__(self, X, y, augment=False, normalize=True, jitter_sigma=0.03):
        X = np.asarray(X, dtype=np.float32)
        _fill_nan(X)
        if normalize:
            X = _zscore(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))
        self.augment     = augment
        self.jitter_sigma = jitter_sigma

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment and self.jitter_sigma > 0:
            x = x + torch.randn_like(x) * self.jitter_sigma
        return x, self.y[idx]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ucr_dataset(name: str, root: str = "./data/ucr"):
    """Load a UCR/UEA time series dataset from .ts files on disk.

    Returns:
        X_train: float32 numpy [N_train, C, T]
        y_train: int64  numpy [N_train]
        X_test:  float32 numpy [N_test,  C, T]
        y_test:  int64  numpy [N_test]
        num_classes: int
        in_ch: int  (number of channels C)
    """
    try:
        from aeon.datasets import load_from_ts_file
    except ImportError as e:
        raise ImportError(
            "aeon is required to load UCR/UEA datasets.  "
            "Install with:  pip install aeon"
        ) from e

    train_path = os.path.join(root, name, f"{name}_TRAIN.ts")
    test_path  = os.path.join(root, name, f"{name}_TEST.ts")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Dataset not found: {train_path}\n"
            f"Download it with:  python tools/download_ts_datasets.py --dataset {name}\n"
            f"or set UCR_ROOT env var to point to your UCR archive directory."
        )

    X_train, y_train = load_from_ts_file(train_path)
    X_test,  y_test  = load_from_ts_file(test_path)

    # aeon returns [N, C, T] float64, or a list of [C, T_i] for variable-length datasets
    if isinstance(X_train, list):
        max_len = max(
            max(x.shape[-1] for x in X_train),
            max(x.shape[-1] for x in X_test),  # type: ignore[union-attr]
        )
        X_train = _pad_ragged(X_train, max_len)
        X_test  = _pad_ragged(X_test, max_len)  # type: ignore[arg-type]
    else:
        X_train = X_train.astype(np.float32)  # type: ignore[union-attr]
        X_test  = X_test.astype(np.float32)   # type: ignore[union-attr]

    y_train, y_test, classes = _encode_labels(y_train, y_test)

    in_ch       = X_train.shape[1]
    num_classes = len(classes)
    print(f"  {name}: {X_train.shape[0]} train / {X_test.shape[0]} test | "
          f"C={in_ch}  T={X_train.shape[2]}  classes={num_classes}")

    return X_train, y_train, X_test, y_test, num_classes, in_ch
