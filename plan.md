# Timeseries Benchmark Experiment Plan

Goal: establish how VNN1D and LaguerreVNN1D compare against standard library baselines
across a multi-dataset benchmark, then use ablations to understand which design choices matter.

Use `--no-wandb` to suppress W&B (logs to files only).

---

## 0. Data setup

Download all benchmark datasets once before running anything:

```bash
python tools/download_ts_datasets.py \
  --root ./data/ucr \
  --dataset ECG5000 FordA ArticularyWordRecognition NATOPS EthanolConcentration \
            UWaveGestureLibrary ElectricDevices
```

---

## 1. Baselines — standard 5-dataset suite

These are the gold-standard CNN baselines from the time-series literature.
Each run covers ECG5000 / FordA / ArticularyWordRecognition / NATOPS / EthanolConcentration.

```bash
# FCN — Wang et al. 2017, ~267K params
python benchmark.py --model fcn --suite standard --wandb_group baselines --no-wandb

# ResNet1D — Wang et al. 2017, ~480K params
python benchmark.py --model resnet1d --suite standard --wandb_group baselines --no-wandb

# InceptionTime — Fawaz et al. 2020, ~457K params
python benchmark.py --model inceptiontime --suite standard --wandb_group baselines --no-wandb
```

Expected runtime: ~30–60 min per model on GPU.

---

## 2. Our models — standard suite (apples-to-apples)

Run VNN1D and LaguerreVNN1D with defaults on the same 5 datasets.

```bash
# VNN1D default (quad + cubic symmetric, base_ch=8, ~200K params)
python benchmark.py --model vnn_1d --suite standard --wandb_group our_models --no-wandb

# LaguerreVNN1D default (degrees=[2,3], base_ch=8, ~94K params)
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group our_models \
  --poly_degrees 2 3 --no-wandb
```

---

## 3. VNN1D ablations — quick suite (ECG5000 + FordA only)

Isolate the effect of each architectural choice. Run on the quick 2-dataset suite
to get results fast; promote winners to the standard suite.

```bash
# A1: Quadratic only — no cubic path
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb \
  --disable_cubic

# A2: Quad + cubic symmetric (default)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb

# A3: Quad + cubic general (a·b·c, more expressive, more params)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb \
  --cubic_mode general

# A4: Lower quadratic rank (Q=1 — minimal nonlinearity, fewer params than default Q=2)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb \
  --Q 1

# A5: Higher quadratic rank (Q=4 instead of default 2)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb \
  --Q 4

# A6: Larger model width (base_ch=12 to match baseline param counts ~200K→~450K)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation --no-wandb \
  --base_ch 12
```

Q sweep (A4→A2→A5) tells you if rank-2 is the right default or if over/under-parameterising matters.

---

## 4. LaguerreVNN1D ablations — quick suite

Explore the polynomial degree and domain scale choices.

```bash
# B1: Degree 1 only — linear Laguerre (sanity check: does any nonlinearity help?)
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 1 --no-wandb

# B2: Degree 2 only — single nonlinear path (quadratic-equivalent)
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 --no-wandb

# B3: Degrees [2, 3] — default, mirrors quad+cubic
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 --no-wandb

# B4: Degrees [2, 3, 4] — three polynomial orders
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 4 --no-wandb

# B5: Degrees [2, 3, 4] with tighter input range (alpha=0.5)
#     Recommended when using degree >= 4 to limit polynomial growth
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 4 --alpha 0.5 --no-wandb

# B6: Higher orders only — skip degree 2, use [3, 4, 5]
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 3 4 5 --alpha 0.5 --no-wandb

# B7: Scaled-up width to match baseline param budget (~450K params at base_ch=16)
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 --base_ch 16 --no-wandb
```

B1→B2→B3→B4 answers "do higher degrees keep helping?"
B4 vs B5 answers "does domain clamping matter at high degree?"

---

## 4.5. Promote winners to standard suite

After quick ablations finish, pick the best VNN config and best Laguerre config and validate
on the full standard 5-dataset suite before committing to the full suite run.

```bash
# Example: if A2 (default) and B3 win on quick suite
python benchmark.py --model vnn_1d --suite standard --wandb_group ablation_winners --no-wandb \
  [best VNN flags]

python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group ablation_winners \
  --poly_degrees [best degrees] [best other flags] --no-wandb
```

---

## 5. Seed runs — 3 seeds for reliable numbers

Single-run results have high variance on small datasets (e.g. NATOPS n=180, EthanolConcentration n=261).
Run each winning config 3 times with different seeds before writing up numbers.

```bash
# Repeat for seed in 0 1 2 — add --seed flag when supported, or re-run manually
for SEED in 0 1 2; do
  python benchmark.py --model <best_model> --suite standard \
    --wandb_group seed_runs --no-wandb [best flags]
done
```

Report mean ± std across seeds, not the single best run.

---

## 6. Best configs — full 7-dataset suite

Once ablations and seed validation are done, run everything on the full suite for the
definitive comparison table.

```bash
# Best baseline (expected: inceptiontime or resnet1d)
python benchmark.py --model inceptiontime --suite full --wandb_group final --no-wandb

# Best VNN1D config (fill in from ablation results)
python benchmark.py --model vnn_1d --suite full --wandb_group final --no-wandb \
  [--Q ? --base_ch ? --cubic_mode ?]

# Best LaguerreVNN1D config (fill in from ablation results)
python benchmark.py --model laguerre_vnn_1d --suite full --wandb_group final --no-wandb \
  --poly_degrees [? ? ?] --alpha [?] --base_ch [?]
```

---

## GPU assignment (4 GPUs — run phases 1–4 simultaneously)

Edit `GPUS=(...)` at the top of `launch_script.sh` to set your physical GPU IDs,
then `bash launch_script.sh`. Total wall time ≈ 60–90 min.

| Variable | Default | Jobs (sequential within GPU) |
|----------|---------|------------------------------|
| `GPUS[0]` | 0 | Phase 1: FCN → ResNet1D → InceptionTime on standard |
| `GPUS[1]` | 1 | Phase 2: VNN1D on standard → Phase 3: ablations A1–A6 on quick |
| `GPUS[2]` | 2 | Phase 2: LaguerreVNN1D on standard → Phase 4: ablations B1–B7 on quick |
| `GPUS[3]` | 3 | Reserved — free for phase 4.5 / seed runs |

After all of the above finish:
- Phase 4.5 on GPU 3: promote winners to standard suite
- Phase 6 on any GPU: seed repeats (3×)
- Phase 6 on any available GPUs: full suite runs (one model per GPU)

---

## What to look for

- **Accuracy vs params**: LaguerreVNN1D at ~94K vs baselines at 267–480K.
  If it's within 2–3% accuracy at 4–5× fewer params, that's a compelling result.
- **B1 vs B2**: Does nonlinearity help at all? (linear Laguerre vs quadratic-only)
- **Which Laguerre degrees help**: B2→B3→B4 progression — when does adding a degree stop helping?
- **Alpha sensitivity**: compare B4 vs B5 to see if domain mapping matters.
- **VNN vs Laguerre**: compare A2 (quad+cubic) vs B3 (deg=[2,3]) — same param budget,
  different interaction mechanism. This is the cleanest ablation of monomial vs orthogonal.
- **Q rank sensitivity**: A4→A2→A5 (Q=1,2,4) — is rank-2 optimal or arbitrary?
- **Seed variance**: datasets with <300 training samples (NATOPS, EthanolConcentration, UWave)
  will have high variance. Don't trust single-run numbers on those.
