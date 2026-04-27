# Timeseries Benchmark — Phase 2.5 (rerun)

Suites expanded; all ablations rerun on the new 10-dataset standard suite.
EthanolConcentration removed permanently (extreme overfitting, uninformative).

---

## Benchmark Suites

| Suite | Datasets | Est. time/model |
|-------|---------|-----------------|
| quick | 2 (ECG5000, FordA) | ~5 min |
| standard | 10 (see below) | ~20–40 min |
| full | 16 (standard + 6) | ~40–80 min |

### Standard suite (10 datasets)

| Dataset | Archive | C | T | Classes | Train |
|---------|---------|--:|--:|--------:|------:|
| ECG5000 | UCR | 1 | 140 | 5 | 500 |
| FordA | UCR | 1 | 500 | 2 | 3601 |
| Wafer | UCR | 1 | 152 | 2 | 1000 |
| ArticularyWordRecognition | UEA | 9 | 144 | 25 | 275 |
| NATOPS | UEA | 24 | 51 | 6 | 180 |
| JapaneseVowels | UEA | 12 | 29 | 9 | 270 |
| Epilepsy | UEA | 3 | 206 | 4 | 137 |
| BasicMotions | UEA | 6 | 100 | 4 | 40 |
| CharacterTrajectories | UEA | 3 | 182 | 20 | 1422 |
| UWaveGestureLibrary | UEA | 3 | 315 | 8 | 120 |

### Full suite adds (6 more)

| Dataset | Archive | C | T | Classes | Train |
|---------|---------|--:|--:|--------:|------:|
| FordB | UCR | 1 | 500 | 2 | 3636 |
| ElectricDevices | UCR | 1 | 96 | 7 | 8926 |
| SpokenArabicDigits | UEA | 13 | 93 | 10 | 6599 |
| Heartbeat | UEA | 61 | 405 | 2 | 204 |
| SelfRegulationSCP1 | UEA | 6 | 896 | 2 | 268 |
| HandMovementDirection | UEA | 10 | 400 | 4 | 160 |

---

## Historical Results (quick suite — ECG5000 / FordA only)

*These ran on the old 2-dataset quick suite. Kept for reference; all are being
rerun on the expanded 10-dataset standard suite in Phase 2.5.*

### All models — quick suite

| Model | Params* | ECG5000 | FordA | Avg |
|-------|--------:|--------:|------:|----:|
| FCN | 265K | 94.0% | 91.5% | 92.8% |
| ResNet1D | 479K | 93.4% | 92.7% | 93.1% |
| InceptionTime | 460K | 93.6% | **96.0%** | **94.8%** |
| VNN default (Q=2, sym) | 50K | 94.0% | 93.0% | 93.5% |
| **C2: VNN Q=1, no cubic** | **36K** | **94.2%** | 94.1% | 94.2% |
| Laguerre [2,3] default | 24K | 93.9% | 93.2% | 93.6% |
| **C4: Laguerre [3,4,5] α=0.5** | **31K** | 94.1% | 93.3% | 93.7% |

### VNN1D ablations — quick suite

| Config | ECG5000 | FordA | Params |
|--------|--------:|------:|-------:|
| A1: no cubic | 93.7% | 93.3% | 50K |
| A2: default (Q=2, sym cubic) | 94.0% | 93.1% | 50K |
| A3: cubic general | 94.0% | 92.7% | 57K |
| A4: Q=1 | 93.7% | 93.6% | 36K |
| A5: Q=4 | 93.8% | 92.8% | 78K |
| A6: ch=12 | 94.0% | 93.1% | 440K |

### Laguerre ablations — quick suite

| Config | ECG5000 | FordA | Params |
|--------|--------:|------:|-------:|
| B1: deg=[1] α=1.0 | 94.2% | 93.4% | 17K |
| B2: deg=[2] α=1.0 | 93.8% | 93.0% | 17K |
| B3: deg=[2,3] α=1.0 | 94.0% | 92.9% | 24K |
| B4: deg=[2,3,4] α=1.0 | 93.7% | 92.9% | 31K |
| B5: deg=[2,3,4] α=0.5 | 93.7% | 93.3% | 31K |
| B6: deg=[3,4,5] α=0.5 | 93.9% | **93.8%** | 31K |
| B7: ch=16 | 93.7% | 92.0% | 366K |

---

## Phase 2.5: Full Ablation Rerun — Expanded Standard Suite (10 datasets)

Re-running every model and ablation config on the new 10-dataset standard suite
to get convincing cross-dataset evidence.  Old quick-suite numbers above should
be treated as preliminary only.

**Download all datasets first:**
```bash
python tools/download_ts_datasets.py \
  --dataset ECG5000 FordA Wafer ArticularyWordRecognition NATOPS \
            JapaneseVowels Epilepsy BasicMotions CharacterTrajectories UWaveGestureLibrary
```

### Baselines

```bash
python benchmark.py --model fcn          --suite standard --wandb_group phase25 --no-wandb
python benchmark.py --model resnet1d     --suite standard --wandb_group phase25 --no-wandb
python benchmark.py --model inceptiontime --suite standard --wandb_group phase25 --no-wandb
```

### VNN1D ablations

```bash
# A1: no cubic
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --disable_cubic

# A2: default (Q=2, symmetric cubic)
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb

# A3: cubic general
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --cubic_mode general

# A4: Q=1
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --Q 1

# A5: Q=4
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --Q 4

# A6: ch=12  (width scaling sanity check)
python benchmark.py --model vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --base_ch 12
```

### Laguerre degree ablations

```bash
# B1: linear-only
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 1 --alpha 1.0

# B2: quadratic-only
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 2 --alpha 1.0

# B3: [2,3] default
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 2 3 --alpha 1.0

# B4: [2,3,4]
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 2 3 4 --alpha 1.0

# B5: [2,3,4] α=0.5
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 2 3 4 --alpha 0.5

# B6: [3,4,5] α=0.5  (quick-suite winner)
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 3 4 5 --alpha 0.5

# D1: [3,4] compact high-degree pair
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 3 4 --alpha 0.5

# D2: [2,3,4,5] four degrees
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5

# D3: [4,5,6] push higher
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase25 --no-wandb \
  --poly_degrees 4 5 6 --alpha 0.5
```

### Phase 2.5 results (fill in after runs)

Columns: ECG=ECG5000, Ford=FordA, Wfr=Wafer, AWR=ArticularyWordRecognition,
NAT=NATOPS, JV=JapaneseVowels, Epi=Epilepsy, BM=BasicMotions,
CT=CharacterTrajectories, UW=UWaveGestureLibrary, Avg=mean of 10.

#### Baselines

| Model | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|-------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| FCN | 265K | — | — | — | — | — | — | — | — | — | — | — |
| ResNet1D | 479K | — | — | — | — | — | — | — | — | — | — | — |
| InceptionTime | 460K | — | — | — | — | — | — | — | — | — | — | — |

#### VNN1D ablations

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| A1: no cubic | 50K | — | — | — | — | — | — | — | — | — | — | — |
| A2: default | 50K | — | — | — | — | — | — | — | — | — | — | — |
| A3: cubic gen | 57K | — | — | — | — | — | — | — | — | — | — | — |
| A4: Q=1 | 36K | — | — | — | — | — | — | — | — | — | — | — |
| A5: Q=4 | 78K | — | — | — | — | — | — | — | — | — | — | — |
| A6: ch=12 | 440K | — | — | — | — | — | — | — | — | — | — | — |

#### Laguerre degree ablations

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| B1: [1] | 17K | — | — | — | — | — | — | — | — | — | — | — |
| B2: [2] | 17K | — | — | — | — | — | — | — | — | — | — | — |
| B3: [2,3] | 24K | — | — | — | — | — | — | — | — | — | — | — |
| B4: [2,3,4] | 31K | — | — | — | — | — | — | — | — | — | — | — |
| B5: [2,3,4] α=0.5 | 31K | — | — | — | — | — | — | — | — | — | — | — |
| B6: [3,4,5] α=0.5 | 31K | — | — | — | — | — | — | — | — | — | — | — |
| D1: [3,4] α=0.5 | 24K | — | — | — | — | — | — | — | — | — | — | — |
| D2: [2,3,4,5] α=0.5 | 38K | — | — | — | — | — | — | — | — | — | — | — |
| D3: [4,5,6] α=0.5 | 31K | — | — | — | — | — | — | — | — | — | — | — |

---

## Phase 3: Seed Runs

Run each winner 3× to get mean ± std.  Run after Phase 2.5 confirms which
configs win on the expanded standard suite — winners may differ from the
quick-suite results above.

```bash
# Repeat 3× for each winner (substitute correct config flags after Phase 2.5)
python benchmark.py --model <winner_model> --suite standard --wandb_group seed_runs --no-wandb [flags]
python benchmark.py --model <winner_model> --suite standard --wandb_group seed_runs --no-wandb [flags]
python benchmark.py --model <winner_model> --suite standard --wandb_group seed_runs --no-wandb [flags]
```

---

## Phase 4: Full 16-dataset Suite

Run the confirmed Phase 3 winners on the full suite.

```bash
python benchmark.py --model inceptiontime --suite full --wandb_group final --no-wandb
python benchmark.py --model <vnn_winner>  --suite full --wandb_group final --no-wandb [flags]
python benchmark.py --model <lag_winner>  --suite full --wandb_group final --no-wandb [flags]
```

**Download full-suite datasets first:**
```bash
python tools/download_ts_datasets.py \
  --dataset FordB ElectricDevices SpokenArabicDigits Heartbeat SelfRegulationSCP1 HandMovementDirection
```

---

## Phase 5: Laguerre Simplification Ablations

Three new model types, each applying one structural simplification to LaguerreVNN1D.
Run on the standard suite against the Phase 2.5 winner for an apples-to-apples comparison.

| Variant | Model name | Change vs base | Params (base_ch=8) |
|---------|-----------|----------------|--------------------|
| Base | `laguerre_vnn_1d` | — | 93K |
| S1 | `laguerre_vnn_1d_s1` | Remove inner clamp on softplus arg | 93K (identical) |
| S2 | `laguerre_vnn_1d_s2` | Shared Conv1d across degrees + learnable α_d | 66K (−29%) |
| S3 | `laguerre_vnn_1d_s3` | Scalar gates instead of per-channel | 93K (−0.5%) |

```bash
# Run after Phase 2.5 — substitute the winning poly_degrees/alpha below
python benchmark.py --model laguerre_vnn_1d    --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s1 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s2 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s3 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 3 4 5 --alpha 0.5
```

### Simplification ablation results (fill in after runs)

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| Base (B6) | 31K | — | — | — | — | — | — | — | — | — | — | — |
| S1: no inner clamp | 31K | — | — | — | — | — | — | — | — | — | — | — |
| S2: shared proj | 22K | — | — | — | — | — | — | — | — | — | — | — |
| S3: scalar gates | 31K | — | — | — | — | — | — | — | — | — | — | — |
