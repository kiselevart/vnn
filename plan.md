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
            JapaneseVowels Epilepsy BasicMotions CharacterTrajectories
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

### Phase 2.5 results

> **Note:** JapaneseVowels and CharacterTrajectories errored (exit code 1) on all
> models — both excluded from averages. Avg = mean over 8 datasets.

Columns: ECG=ECG5000, Ford=FordA, Wfr=Wafer, AWR=ArticularyWordRecognition,
NAT=NATOPS, JV=JapaneseVowels, Epi=Epilepsy, BM=BasicMotions,
CT=CharacterTrajectories, Avg=mean of 8 (JV+CT excluded).

#### Baselines

| Model | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|-------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| FCN | 265K | 94.1 | 92.3 | 99.4 | 99.3 | 89.4 | ERR | 99.3 | 100.0 | ERR | 84.4 | 94.8 |
| ResNet1D | 479K | 93.7 | 92.0 | 99.6 | 98.7 | 88.3 | ERR | 98.5 | 100.0 | ERR | 84.1 | 94.4 |
| InceptionTime | 460K | 93.7 | 95.5 | 99.8 | 99.3 | 88.3 | ERR | 97.1 | 100.0 | ERR | 89.7 | **95.4** |

#### VNN1D ablations

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| A1: no cubic | 50K | 94.2 | 93.6 | 99.3 | 98.3 | 87.2 | ERR | 98.5 | 100.0 | ERR | 88.4 | **94.9** |
| A2: default | 50K | 94.0 | 93.3 | 99.6 | 98.3 | 87.2 | ERR | 96.4 | 100.0 | ERR | 87.8 | 94.6 |
| A3: cubic gen | 57K | 93.7 | 92.8 | 99.5 | 98.7 | 86.1 | ERR | 97.1 | 100.0 | ERR | 87.2 | 94.4 |
| A4: Q=1 | 36K | 93.9 | 93.3 | 99.2 | 99.0 | 87.2 | ERR | 97.1 | 95.0 | ERR | 85.0 | 93.7 |
| A5: Q=4 | 78K | 93.9 | 92.3 | 99.5 | 99.0 | 88.3 | ERR | 99.3 | 100.0 | ERR | 88.1 | **95.1** |
| A6: ch=12 | 440K | 93.8 | 92.8 | 99.7 | 98.3 | 86.1 | ERR | 100.0 | 100.0 | ERR | 88.1 | 94.9 |

#### Laguerre degree ablations

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| B1: [1] | 17K | 93.8 | 93.2 | 99.5 | 97.7 | 85.6 | ERR | 96.4 | 97.5 | ERR | 86.2 | 93.7 |
| B2: [2] | 17K | 93.7 | 93.0 | 99.5 | 97.0 | 85.6 | ERR | 96.4 | 100.0 | ERR | 86.9 | 94.0 |
| B3: [2,3] | 24K | 93.8 | 93.3 | 99.2 | 98.7 | 87.2 | ERR | 96.4 | 100.0 | ERR | 84.1 | 94.1 |
| B4: [2,3,4] | 31K | 94.0 | 93.0 | 99.6 | 99.0 | 87.2 | ERR | 99.3 | 100.0 | ERR | 83.8 | 94.5 |
| B5: [2,3,4] α=0.5 | 31K | 94.1 | 93.7 | 99.3 | 99.3 | 83.9 | ERR | 96.4 | 97.5 | ERR | 84.4 | 93.6 |
| B6: [3,4,5] α=0.5 | 31K | 94.2 | 93.4 | 99.4 | 99.3 | 81.7 | ERR | 97.8 | 100.0 | ERR | 86.6 | 94.1 |
| D1: [3,4] α=0.5 | 24K | 93.7 | 93.9 | 99.3 | 99.0 | 83.3 | ERR | 97.8 | 100.0 | ERR | 83.8 | 93.9 |
| D2: [2,3,4,5] α=0.5 | 38K | 93.7 | 92.9 | 99.5 | 98.3 | 87.8 | ERR | 98.5 | 100.0 | ERR | 86.9 | **94.7** |
| D3: [4,5,6] α=0.5 | 31K | 94.0 | 92.4 | 99.5 | 99.0 | 84.4 | ERR | 95.7 | 100.0 | ERR | 87.5 | 94.1 |

#### Winners

- **VNN winner: A1 (no cubic, 50K)** — 94.9% avg; cubic terms add no value and hurt on average
- **VNN runner-up: A5 (Q=4, 78K)** — 95.1% avg (best VNN, but 1.6× more params than A1)
- **Laguerre winner: D2 ([2,3,4,5] α=0.5, 38K)** — 94.7% avg
- **Best baseline: InceptionTime** — 95.4% avg
- **Key finding:** B6 was the quick-suite winner but NATOPS collapses to 81.7% on the full suite; α=1.0 (B4) and mixed-degree (D2) are more robust

---

## Phase 3: Seed Runs

Winners from Phase 2.5 (8-dataset avg, JV+CT excluded):

| Model | Params | Avg |
|-------|-------:|----:|
| InceptionTime | 460K | 95.4% |
| A5: VNN Q=4 | 78K | 95.1% |
| A1: VNN no-cubic | 50K | 94.9% |
| D2: Laguerre [2,3,4,5] α=0.5 | 38K | 94.7% |

Run each 3× to get mean ± std.  See `launch_script.sh` for the Phase 3 run script.

```bash
# InceptionTime ×3
python benchmark.py --model inceptiontime --suite standard --wandb_group phase3_seeds --no-wandb
python benchmark.py --model inceptiontime --suite standard --wandb_group phase3_seeds --no-wandb
python benchmark.py --model inceptiontime --suite standard --wandb_group phase3_seeds --no-wandb

# VNN A1 (no cubic) ×3
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --disable_cubic
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --disable_cubic
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --disable_cubic

# VNN A5 (Q=4) ×3
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --Q 4
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --Q 4
python benchmark.py --model vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb --Q 4

# Laguerre D2 ([2,3,4,5] α=0.5) ×3
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group phase3_seeds --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5
```

### Phase 3 results (mean over 3 seeds, 8-dataset avg, JV+CT excluded)

| Model | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|-------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| InceptionTime | 460K | 93.5 | 95.3 | 99.8 | 99.2 | 89.6 | ERR | 97.8 | 98.3 | ERR | 89.9 | **95.4±0.2** |
| A1: VNN no-cubic | 50K | 93.9 | 93.6 | 99.4 | 97.9 | 86.5 | ERR | 97.6 | 99.2 | ERR | 86.3 | **94.3±0.2** |
| D2: Laguerre [2,3,4,5] α=0.5 | 38K | 93.9 | 93.4 | 99.5 | 99.1 | 83.1 | ERR | 97.8 | 99.2 | ERR | 87.5 | **94.2±0.2** |
| A5: VNN Q=4 | 78K | 93.7 | 92.8 | 99.5 | 98.7 | 84.8 | ERR | 97.8 | 96.7 | ERR | 86.6 | 93.8±0.5 |

**Key findings:**
- A5 was a lucky single seed in Phase 2.5 (95.1% → 93.8% mean); dropped from Phase 4
- A1 (94.3±0.2) and D2 (94.2±0.2) are essentially tied; D2 wins on params (38K vs 50K)
- NATOPS is the highest-variance dataset across seeds (IT range: 87.2–91.1%, A5: 80.6–88.9%)
- **Phase 4 forward: IT, A1, D2**

---

## Phase 4: Full 16-dataset Suite

> **Prerequisite:** JapaneseVowels and CharacterTrajectories still error (exit 1)
> on the standard suite. Investigate and fix before launching Phase 4, or accept
> a 14-dataset effective suite.

Confirmed winners from Phase 3: InceptionTime, A1 (VNN no-cubic), D2 (Laguerre [2,3,4,5] α=0.5).
Phase 5 simplification ablations run in parallel on the standard suite (same launch, GPU D).
See `launch_script.sh` for the combined Phase 4+5 run.

**Download full-suite datasets first:**
```bash
python tools/download_ts_datasets.py \
  --dataset FordB ElectricDevices SpokenArabicDigits Heartbeat SelfRegulationSCP1 HandMovementDirection
```

```bash
python benchmark.py --model inceptiontime   --suite full --wandb_group phase4 --no-wandb
python benchmark.py --model vnn_1d          --suite full --wandb_group phase4 --no-wandb --disable_cubic
python benchmark.py --model laguerre_vnn_1d --suite full --wandb_group phase4 --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5
```

### Phase 4 results (fill in after runs)

| Model | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | FordB | ElecDev | SAD | HB | SCP1 | HMD | Avg |
|-------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|------:|--------:|----:|---:|-----:|----:|----:|
| InceptionTime | 460K | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| A1: VNN no-cubic | 50K | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| D2: Laguerre D2 | 38K | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

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
# Base config: D2 winner ([2,3,4,5] α=0.5) — Phase 2.5 Laguerre winner
python benchmark.py --model laguerre_vnn_1d    --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s1 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s2 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5

python benchmark.py --model laguerre_vnn_1d_s3 --suite standard --wandb_group simp_ablations --no-wandb \
  --poly_degrees 2 3 4 5 --alpha 0.5
```

### Simplification ablation results (fill in after runs)

| Config | Params | ECG | Ford | Wfr | AWR | NAT | JV | Epi | BM | CT | UW | Avg |
|--------|-------:|----:|-----:|----:|----:|----:|---:|----:|---:|---:|---:|----:|
| Base (D2) | 38K | — | — | — | — | — | — | — | — | — | — | — |
| S1: no inner clamp | 38K | — | — | — | — | — | — | — | — | — | — | — |
| S2: shared proj | 27K | — | — | — | — | — | — | — | — | — | — | — |
| S3: scalar gates | 38K | — | — | — | — | — | — | — | — | — | — | — |
