# W&B Data Explorer

`pull_wandb.py` fetches W&B summaries into `runs.parquet`. `explorer.py` is the
interactive layer for selecting named run collections and plotting metrics.

`prop2_gradient_validation.ipynb` is a focused notebook for visualizing
per-block gradient metrics as empirical support for Proposition 2.

## Refresh Cached Runs

```bash
cd vnn
python wandb_data/pull_wandb.py --no-history --force
```

Fetch histories only when you want fully offline curve plotting:

```bash
python wandb_data/pull_wandb.py --force
```

## Interactive Usage

Run from the `vnn/` directory:

```python
from wandb_data.explorer import WandbExplorer

ex = WandbExplorer()

legacy = ex.collect(
    "VNN legacy",
    name_contains="legacy_fusion_Q",
    dataset="ucf101",
)
r3d = ex.collect(
    "R3D",
    name_contains="r3d",
    dataset="ucf101",
)

legacy.table()
ex.summary([legacy, r3d], metric="test_acc")
ex.plot_summary([legacy, r3d], metric="test_acc")
ex.plot_history([legacy, r3d], metric="val_acc", smooth=3)
```

Useful metrics:

- Summary: `test_acc`, `test_acc5`, `val_acc`, `val_loss`, `train_acc`, `train_loss`
- History: `val_acc` / `val/acc`, `val_loss` / `val/loss`, `train_acc`,
  `train_loss`, `train_grad_norm`

Useful selectors:

- `name_contains="vnn_legacy_fusion"`
- `name_contains=["vnn_additive_fusion_ho", "no_cubic"]`
- `model="small_r3d"`
- `group="small_baselines"`
- `dataset="ucf101"`
- `split=[1, 2, 3]`
- `query="test_acc.notna() and Q <= 4"`
