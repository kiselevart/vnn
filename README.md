
Code release for [Volterra Neural Networks (VNNs)](https://arxiv.org/abs/1910.09616) and [Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/6870).

Patent Information: [Volterra Neural Network and Method](https://patents.google.com/patent/US20210279519A1/en?q=(siddharth+roheda)&oq=siddharth+roheda)

# Citation
If you use our work please cite and acknowledge:

@inproceedings{roheda2020conquering,<br />
  title={Conquering the cnn over-parameterization dilemma: A volterra filtering approach for action recognition},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},<br />
  volume={34},<br />
  number={07},<br />
  pages={11948--11956},<br />
  year={2020}<br />
}<br />

@article{roheda2019volterra,<br />
  title={Volterra Neural Networks (VNNs)},<br />
  author={Roheda, Siddharth and Krim, Hamid},<br />
  journal={arXiv preprint arXiv:1910.09616},<br />
  year={2019}<br />
}<br />

---

# Volterra Neural Networks — Extended Implementation

This repository extends the original VNN codebase with a research implementation focused on **video action recognition** (UCF101, HMDB51, UCF10, UCF11) and **image classification** (CIFAR-10). The core idea is replacing standard convolutions with 2nd- and 3rd-order polynomial (Volterra series) interactions to reduce over-parameterization while capturing richer feature interactions.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Models](#models)
- [Architecture](#architecture)
- [Volterra Math](#volterra-math)
- [Training Arguments](#training-arguments)
- [Weights & Biases Logging](#weights--biases-logging)
- [Resuming Training](#resuming-training)
- [Project Structure](#project-structure)

---

## Overview

A standard convolution computes a *linear* weighted sum of input features. Volterra series extend this to polynomial interactions:

```
y = h0 + Σ h1(i)·x(i)              # linear (1st order)
       + Σ h2(i,j)·x(i)·x(j)       # quadratic (2nd order)
       + Σ h3(i,j,k)·x(i)·x(j)·x(k) # cubic (3rd order)
```

Naively computing all cross-terms is intractable (O(C²) for quadratic, O(C³) for cubic). This implementation uses CP-style tensor factorizations to keep the parameter count tractable while preserving the expressive power of higher-order interactions.

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2.0 is required for mixed-precision (AMP) support and `torch.compile` compatibility. For a specific CUDA version (e.g., CUDA 11.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Dataset Setup

Dataset paths are resolved by `mypath.py` in the following priority order:

1. **Environment variable** (preferred for server use)
2. **Repository-local default** under `./data/`

### Environment Variables

```bash
# UCF101
export UCF101_ROOT=/path/to/ucf101          # raw video directory
export UCF101_PREPROCESSED=/path/to/ucf101_pre  # extracted frames (auto-created on first run)

# HMDB51
export HMDB51_ROOT=/path/to/hmdb51
export HMDB51_PREPROCESSED=/path/to/hmdb51_pre
```

### Video Directory Structure

Video datasets must follow this layout:

```
ucf101/
  train/
    ApplyEyeMakeup/
      v_ApplyEyeMakeup_g01_c01.avi
      ...
    BaseballPitch/
      ...
  val/
    ...
  test/
    ...
```

Frame extraction runs automatically on the first training call and is cached in the preprocessed directory. Optical flow for fusion models is computed on-the-fly by default (no pre-extraction needed).

Preprocessed frames are split-aware: the first run for a given split writes to `ucf101_pre/split1/`, `ucf101_pre/split2/`, etc., so multiple splits coexist on disk.

---

## Training

### Single GPU

All single-GPU training goes through `train.py`:

```bash
# CIFAR-10
python3 train.py --dataset cifar10 --model vnn_ortho --epochs 50 --batch_size 128 --lr 0.01

# Video — RGB only
python3 train.py --dataset ucf101 --model vnn_rgb_ho --epochs 100 --batch_size 8 --lr 1e-4 --num_workers 8

# Video — Two-stream fusion (RGB + optical flow)
python3 train.py --dataset ucf101 --model vnn_fusion_ho --epochs 100 --batch_size 8 --lr 1e-4 --num_workers 8
```

### Multi-GPU (DDP)

Use `train_par.py` with `torchrun` for multi-GPU training. Scale `--lr` linearly with the number of GPUs (1 GPU → `1e-4`, 4 GPUs → `4e-4`):

```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29500 train_par.py \
    --dataset ucf101 \
    --model vnn_fusion_ho \
    --epochs 100 \
    --batch_size 8 \
    --lr 4e-4 \
    --num_workers 8 \
    --run_name vnn_fusion_ucf101
```

`train_par.py` accepts the same arguments as `train.py`.

### I3D Two-Stream

I3D two-stream training uses a dedicated script that handles DDP internally:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python tools/train_i3d_two_stream.py \
    --dataset ucf101 --run_name i3d_twostream_ucf101

# 4-GPU DDP
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29502 tools/train_i3d_two_stream.py \
    --dataset ucf101 --run_name i3d_twostream_ucf101 --lr 4e-4
```

### Dataset Splits

UCF101 and HMDB51 have three official train/test splits. Use `--split` to select:

```bash
python3 train.py --dataset ucf101 --model vnn_fusion_ho --split 2 --run_name vnn_ucf101_split2 ...
```

---

## Evaluation

Run evaluation only (no training) on a saved checkpoint:

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --test_only \
  --resume runs/<run_name>/checkpoints/best_model.pth
```

---

## Models

### VNN Models (via `train.py` / `train_par.py`)

| Model name | Task | Description |
|---|---|---|
| `vnn_ortho` | CIFAR-10 | ResNet18-like backbone with Chebyshev T2(x) polynomial features and spectral normalization |
| `vnn_simple` | CIFAR-10 | Lightweight VNN without orthogonalization |
| `resnet18` | CIFAR-10 | Standard ResNet18 baseline (adapted for 32×32 input) |
| `vnn_rgb_ho` | Video | 4-block 3D backbone (quadratic + cubic) → fusion classifier head, RGB only; use `--disable_cubic` to ablate cubic |
| `vnn_fusion_ho` | Video | Two-stream: RGB backbone + flow backbone → cross-stream product → fusion head; use `--disable_cubic` to ablate cubic |
| `vnn_additive_fusion_ho` | Video | Additive fusion ablation: `cat(rgb, flow)` only — no cross-stream product; isolates the rgb×flow interaction |
| `vnn_legacy_fusion` | Video | Legacy arch matching original paper (no gates, no shortcuts, no cubic, additive fusion); use `--no_amp` |
| `vnn_legacy_rgb` | Video | Legacy arch, RGB-only single-stream variant; use `--no_amp` |
| `vnn_rgb` | Video | Legacy RGB-only model (older backbone) |
| `vnn_fusion` | Video | Legacy two-stream fusion (older backbone) |

### Orthogonal Polynomial Basis Models (via `train.py` / `train_par.py`)

Models that parameterise the temporal dimension of each Conv3d kernel as a linear combination of orthogonal basis functions. All share the same 4-block backbone topology as the Laguerre models and differ only in their temporal basis:

| Model name | Temporal basis | Description |
|---|---|---|
| `lvn_laguerre_rgb` / `lvn_laguerre_fusion` | Laguerre | Exponential decay toward past frames; causal/asymmetric bias |
| `lvn_legendre_rgb` / `lvn_legendre_fusion` | Legendre | Uniform temporal weighting; equal attention to all frames |
| `lvn_chebyshev_rgb` / `lvn_chebyshev_fusion` | Chebyshev T₁ | Same uniform domain as Legendre; better approximation for same N |
| `lvn_hermite_rgb` / `lvn_hermite_fusion` | Hermite | Gaussian envelope centred at mid-clip; symmetric past/future decay |

Use `--n_lag N` to set the number of basis functions (temporal compression), `--alpha` to control scale (Laguerre/Hermite only). Omitting `--n_lag` uses full expressiveness (N = kernel T). The LVN/TLVN/orthogonal `_fusion` variants use additive two-stream fusion (`cat(rgb, flow)`) without the explicit cross-stream product.

```bash
python3 train_par.py --dataset ucf101 --model lvn_legendre_fusion \
    --n_lag 4 --run_name legendre_ucf101 --lr 4e-4
```

### Baseline Models (via `train.py` / `train_par.py`)

| Model name | Task | Description |
|---|---|---|
| `r3d` | Video | R3D-18: 3D ResNet with pure Conv3D spatiotemporal blocks |
| `r2plus1d` | Video | R(2+1)D-18: factorized spatiotemporal convolutions (spatial 2D + temporal 1D) |
| `resnet50_frame_avg` | Video | ResNet-50 applied per-frame; logits averaged across frames (no temporal modelling) |

### I3D Two-Stream (via `tools/train_i3d_two_stream.py`)

Inflated 3D Inception (I3D) with separate RGB and optical flow streams, late-fused by averaging logits. Auxiliary classifiers are used during training.

---

## Architecture

### Video Model Data Flow

```
Input RGB:  [B, 3, T, H, W]
Input Flow: [B, 2, T, H, W]   ← computed on-the-fly for fusion models
     │                │
     ▼                ▼
 backbone_4block   backbone_4block
     │                │
  [B,96,T/8,H/8,W/8]  [B,96,T/8,H/8,W/8]
     │                │
     └────── × ───────┘   ← element-wise cross-stream product (vnn_fusion_ho)
     │        │       │
   rgb       flow   cross
     └────── cat ────┘
          [B, 288, T/8, H/8, W/8]
               │
          fusion_head
          (VolterraBlock3D → ClassifierHead)
               │
          [B, num_classes]
```

For `vnn_additive_fusion_ho`, the cross-stream product is omitted and only `cat(rgb, flow)` is passed to the fusion head.

### 4-Block Backbone (`backbone_4block.py`)

```
Block 1: MultiKernelBlock3D  — parallel 5×5×5 + 3×3×3 + 1×1×1 convs, quadratic only, stride=2
         → [B, 24, T/2, H/2, W/2]

Block 2: VolterraBlock3D     — quadratic, stride=2, residual shortcut
         → [B, 32, T/4, H/4, W/4]

Block 3: VolterraBlock3D     — quadratic + symmetric cubic, no pool, residual shortcut
         → [B, 64, T/4, H/4, W/4]

Block 4: VolterraBlock3D     — quadratic + symmetric cubic, stride=2, residual shortcut
         → [B, 96, T/8, H/8, W/8]
```

### VolterraBlock3D

Each block has three parallel paths summed together:

```
x ──┬── conv_lin  ──→ BN ──────────────────────────────────┐
    ├── conv_quad ──→ BN → volterra_quadratic → quad_gate ──┤→ sum → out
    └── conv_cub  ──→ BN → volterra_cubic_sym → cubic_gate ─┘

Residual shortcut: x → 1×1×1 conv (if in_ch ≠ out_ch or stride > 1) → added to out
```

Gates are per-channel scalar parameters initialized at `1e-4`, easing in polynomial contributions gradually during training.

### Fusion Head (`fusion_head.py`)

A single `VolterraBlock3D` followed by `ClassifierHead`:

```
[B, C, T, H, W] → VolterraBlock3D(stride=2) → AdaptiveAvgPool → Dropout → FC → [B, num_classes]
```

The FC layer receives a 10× learning rate multiplier relative to the backbone.

### CIFAR Model (`cifar_ortho/res_vnn_ortho.py`)

ResNet18-equivalent with Chebyshev polynomial second-order features:

```
T2(x, y) = 4xy − 2   (Chebyshev T2 of the first kind)
```

Spectral normalization on conv weights provides Lipschitz stability. Uses SGD + Nesterov + CosineAnnealingLR.

---

## Volterra Math

Each order is factorized using CP (CANDECOMP/PARAFAC) decomposition to keep parameter counts tractable. All ops clamp outputs to `[-50, 50]` before summation for numerical stability.

### 2nd-order (Quadratic)

```
h2(i,j) ≈ Σ_q a_q(i) · b_q(j)

Implementation:
  conv_quad: Conv3d(in_ch, 2·Q·out_ch, ...)
  split into left [B, Q·C, T, H, W] and right [B, Q·C, T, H, W]
  product = left * right   → clamp → sum over Q → [B, C, T, H, W]
```

### 3rd-order Symmetric

```
h3(i,j,k) ≈ Σ_q a_q(i) · a_q(j) · b_q(k)

Implementation:
  conv_cub: Conv3d(in_ch, 2·Qc·out_ch, ...)
  product = a² * b   → clamp → sum over Qc → [B, C, T, H, W]

The a² term is an energy/magnitude detector (always ≥ 0); b modulates sign and scale.
```

### 3rd-order General (available via `--cubic_mode general`)

```
h3(i,j,k) ≈ Σ_q a_q(i) · b_q(j) · c_q(k)

Implementation:
  conv_cub: Conv3d(in_ch, 3·Qc·out_ch, ...)
  product = a * b * c   → clamp → sum over Qc → [B, C, T, H, W]

More expressive than symmetric but uses 50% more parameters for the cubic path.
```

The `--Q` flag sets the quadratic rank for legacy models (default: 2). In the 4-block backbone (`vnn_fusion_ho`, `vnn_rgb_ho`), Q is hardcoded at 4 per block and `--Q` has no effect.

---

## Training Arguments

The following arguments apply to both `train.py` and `train_par.py` unless noted.

| Argument | Default | Description |
|---|---|---|
| `--dataset` | required | `cifar10`, `ucf10`, `ucf11`, `ucf101`, `hmdb51` |
| `--model` | required | Model name (see Models table) |
| `--run_name` | required | Run directory name (used for checkpoints and W&B display) |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size per GPU |
| `--lr` | 1e-4 | Base learning rate. Scale linearly with number of GPUs for DDP. |
| `--weight_decay` | 1e-4 | L2 regularization strength |
| `--label_smoothing` | 0.1 | Cross-entropy label smoothing factor |
| `--split` | 1 | UCF101/HMDB51 official split number (1–3) |
| `--Q` | 2 | Quadratic rank (CP components). Only affects legacy and LVN models. |
| `--disable_cubic` | False | Remove cubic path from `vnn_rgb_ho` / `vnn_fusion_ho` (quadratic only) |
| `--cubic_mode` | `symmetric` | Cubic factorization: `symmetric` (a²b) or `general` (abc) |
| `--clip_len` | 16 | Frames per clip |
| `--num_workers` | 16 | DataLoader worker processes |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--no_amp` | False | Disable AMP (float16). Required for legacy models without output clamping. |
| `--seed` | None | Global random seed for reproducibility |
| `--resume` | None | Path to checkpoint `.pth` to resume from |
| `--test_only` | False | Skip training, run evaluation on the test set only |
| `--no_wandb` | False | Disable W&B logging entirely |
| `--wandb_group` | required | W&B group name for grouping multi-split or multi-seed runs |

---

## Weights & Biases Logging

W&B is the default logging backend. Login once per machine:

```bash
wandb login
```

Then run training normally — the run name is set via `--run_name`.

**Disable W&B:**
```bash
python3 train.py --dataset ucf101 --model vnn_fusion_ho --run_name exp --no_wandb
```

**Group multi-split runs in W&B:**
```bash
for split in 1 2 3; do
  python3 train.py --dataset ucf101 --model vnn_fusion_ho \
    --split $split --run_name vnn_ucf_split${split} \
    --wandb_group vnn_ucf101_3split
done
```

Logged metrics: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `train/grad_norm`, `lr`.

---

## Resuming Training

Checkpoints are saved to `runs/<run_name>/checkpoints/best_model.pth` (best validation accuracy) and `last_model.pth` (final epoch).

```bash
python3 train.py \
  --dataset ucf101 \
  --model vnn_fusion_ho \
  --epochs 100 \
  --resume runs/vnn_fusion_ho_ucf101_<timestamp>/checkpoints/best_model.pth
```

The optimizer and scheduler states are restored from the checkpoint. Training continues from where it left off.

---

## Numerical Stability

Several mechanisms guard against the instability inherent in polynomial feature interactions:

- **Output clamping:** All Volterra products are clamped to `[-50, 50]` before the rank-sum reduction.
- **Gate initialization:** Quadratic and cubic gates are initialized at `1e-4`, making the network behave like a pure linear model at the start of training. Gates grow gradually as training proceeds.
- **Gradient clipping:** Gradient norm is clipped to 1.0 each step.
- **Batch skipping:** Batches with non-finite loss or non-finite model outputs are skipped rather than propagating NaN gradients.
- **Mixed precision:** AMP (float16) is used automatically on CUDA; the GradScaler handles underflow/overflow. Use `--no_amp` for legacy models that lack clamping.

---

## Project Structure

```
.
├── train.py                        # Single-GPU training entrypoint (Trainer class)
├── train_par.py                    # Multi-GPU DDP training entrypoint (same args as train.py)
├── mypath.py                       # Dataset path resolver (env vars + defaults)
├── requirements.txt
│
├── tools/
│   ├── train_i3d_two_stream.py     # I3D two-stream trainer (DDP-capable, separate entrypoint)
│   └── finished_scripts/           # Completed experiment scripts (archived for reference)
│
├── network/
│   ├── cifar/
│   │   └── vnn_cifar.py            # Simple VNN for CIFAR-10
│   ├── cifar_ortho/
│   │   └── res_vnn_ortho.py        # ResNet18-like VNN with Chebyshev T2 + spectral norm
│   ├── video/                      # Video baselines and I3D
│   │   ├── i3d.py                  # I3D + I3DTwoStream (with auxiliary classifiers)
│   │   ├── established_models.py   # R3DNet, R2Plus1DNet, ResNet50FrameAvg
│   │   ├── vnn_rgb_of_highQv2.py   # Legacy RGB backbone
│   │   └── vnn_fusion_highQv2.py   # Legacy fusion head
│   └── video_higher_order/         # Current primary video model family
│       ├── volterra_blocks.py      # Math primitives: quadratic, cubic_symmetric, cubic_general
│       ├── vnn_4block.py           # VNNFusionHO, VNNAdditiveFusionHO, VNNRgbHO
│       └── vnn_legacy.py           # VNNLegacyFusion, VNNLegacyRgb
│
├── utils/
│   ├── model_factory.py            # Instantiates all models by name string
│   └── data_factory.py             # FlowDatasetWrapper; wraps VideoDataset with optical flow
│
├── dataloaders/
│   └── dataset.py                  # VideoDataset with group-disjoint UCF101/HMDB51 splitting
│
├── data/                           # Default dataset root (gitignored)
└── runs/                           # Training outputs: checkpoints, W&B files (gitignored)
```
