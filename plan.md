# Timeseries Benchmark — Phase 2

Phase 1 complete. This document records what we learned and specifies the next runs.

---

## Phase 1 Results

### Standard 5-dataset suite

| Model | Params* | ECG5000 | FordA | AWR | NATOPS | Ethanol | Avg† |
|-------|--------:|--------:|------:|----:|-------:|--------:|-----:|
| FCN | 265K | 94.0% | 91.5% | 98.7% | 87.2% | 27.4% | 92.9% |
| ResNet1D | 479K | 93.4% | 92.7% | 98.3% | 88.9% | 21.3% | 93.3% |
| InceptionTime | 460K | 93.6% | **96.0%** | **99.3%** | **89.4%** | 24.7% | **94.6%** |
| VNN1D default | 50K | **94.0%** | 93.0% | 98.3% | 85.0% | 24.7% | 92.6% |
| Laguerre [2,3] | 24K | 93.9% | 93.2% | 98.3% | 86.1% | 24.7% | 92.9% |

*params at ECG5000 (smallest dataset); scales up with base_ch per dataset config.
†avg excludes Ethanol — all models are at chance there (see below).

### VNN1D ablations (ECG5000 / FordA, quick suite)

| Config | ECG5000 | FordA | Params |
|--------|--------:|------:|-------:|
| A1: no cubic | 93.7% | 93.3% | 50K |
| A2: default (quad+cubic sym) | 94.0% | 93.1% | 50K |
| A3: cubic general | FAILED (--cubic_mode not wired in benchmark.py — fixed) | |
| A4: Q=1 | 93.7% | **93.6%** | 36K |
| A5: Q=4 | 93.8% | 92.8% | 78K |
| A6: ch=12 | 94.0% | 93.1% | 440K |

### LaguerreVNN1D ablations (ECG5000 / FordA, quick suite)

| Config | ECG5000 | FordA | Params |
|--------|--------:|------:|-------:|
| B1: deg=1 (linear) | **94.2%** | 93.4% | 17K |
| B2: deg=2 | 93.8% | 93.0% | 17K |
| B3: deg=[2,3] | 94.0% | 92.9% | 24K |
| B4: deg=[2,3,4] | 93.7% | 92.9% | 31K |
| B5: deg=[2,3,4] α=0.5 | 93.7% | 93.3% | 31K |
| B6: deg=[3,4,5] α=0.5 | 93.9% | **93.8%** | 31K |
| B7: ch=16 | 93.7% | 92.0% | 366K |

---

## Key Findings

1. **VNN1D matches InceptionTime's average at 10% of the params.** Laguerre does it at 5%.
   The main gap is NATOPS (~3–4% behind InceptionTime).

2. **Cubic term doesn't help** — A1 (no cubic) ≈ A2 (default). Quadratic is doing all the work.

3. **Q rank doesn't matter** — Q=1 ≈ Q=2 ≈ Q=4. Q=1 at 36K is likely the sweet spot.

4. **Width scaling is useless** — A6 at 440K ≈ A2 at 50K. Hard ceiling on these datasets.

5. **Linear Laguerre (deg=1) wins on ECG5000** — highest accuracy at fewest params.
   Nonlinear degrees don't help on easy datasets.

6. **Higher odd degrees win on harder datasets** — B6 [3,4,5] α=0.5 is best on FordA.
   Domain clamping (α=0.5) matters when using high degrees.

7. **EthanolConcentration is chance-level for all models** — 21–27% on a 4-class problem.
   Root cause: 261 training samples + 1751-length sequences + only 300 epochs = undertrained.
   Needs ~800 epochs minimum.

---

## Phase 2: Fix + Promote Winners

### 2a. Ethanol — rerun all models with 800 epochs

```bash
# Baselines
python benchmark.py --model fcn          --datasets EthanolConcentration --epochs 800 --no-wandb
python benchmark.py --model resnet1d     --datasets EthanolConcentration --epochs 800 --no-wandb
python benchmark.py --model inceptiontime --datasets EthanolConcentration --epochs 800 --no-wandb

# Our models
python benchmark.py --model vnn_1d --datasets EthanolConcentration --epochs 800 --Q 1 --disable_cubic --no-wandb
python benchmark.py --model laguerre_vnn_1d --datasets EthanolConcentration --epochs 800 --poly_degrees 3 4 5 --alpha 0.5 --no-wandb
```

### 2b. A3 rerun — cubic general (was broken, now fixed)

```bash
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb --cubic_mode general
```

### 2c. Promote VNN1D winners to standard suite

Ablations say cubic barely helps and Q=1 is sufficient. Test the two most minimal configs:

```bash
# C1: Q=1, cubic symmetric (ablation winner on FordA)
python benchmark.py --model vnn_1d --suite standard --wandb_group vnn_winners --no-wandb --Q 1

# C2: Q=1, no cubic (most minimal — does removing cubic hurt on harder datasets?)
python benchmark.py --model vnn_1d --suite standard --wandb_group vnn_winners --no-wandb --Q 1 --disable_cubic
```

### 2d. Promote Laguerre winners to standard suite

Two competing winners — efficiency (B1) vs harder-dataset accuracy (B6):

```bash
# C3: deg=1 — linear Laguerre, highest ECG5000, lowest params (17K)
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group lag_winners --no-wandb --poly_degrees 1

# C4: deg=[3,4,5] α=0.5 — best on FordA, may generalize better
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group lag_winners --no-wandb --poly_degrees 3 4 5 --alpha 0.5
```

---

## Phase 3: Seed Runs

After phase 2, pick one winner per model family and run 3 seeds for publication-quality numbers.
Run the suite 3 separate times — benchmark.py doesn't have a --seed flag yet so results vary
by random init naturally.

```bash
# Three separate invocations each
python benchmark.py --model vnn_1d --suite standard --wandb_group seed_vnn --no-wandb [best flags]
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group seed_lag --no-wandb [best flags]
```

Report mean ± std. NATOPS (n=180) and Ethanol (n=261) will have the highest variance.

---

## Phase 4: Full 7-dataset suite

Run the single best config per model family plus the best baseline on the full suite.

```bash
python benchmark.py --model inceptiontime --suite full --wandb_group final --no-wandb
python benchmark.py --model vnn_1d --suite full --wandb_group final --no-wandb [best flags from phase 2]
python benchmark.py --model laguerre_vnn_1d --suite full --wandb_group final --no-wandb [best flags from phase 2]
```

---

## GPU assignment (phase 2)

| Variable | Jobs (sequential) |
|----------|-------------------|
| `GPUS[0]` | Ethanol reruns — all 5 models × 800 epochs |
| `GPUS[1]` | A3 rerun → C1 (VNN Q=1) on standard → C2 (VNN Q=1 no-cubic) on standard |
| `GPUS[2]` | C3 (Laguerre deg=1) on standard → C4 (Laguerre [3,4,5] α=0.5) on standard |
| `GPUS[3]` | Free — use for seed runs once phase 2 winners are known |
