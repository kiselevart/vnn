# Timeseries Benchmark Experiment Plan

Goal: establish how VNN1D and LaguerreVNN1D compare against standard library baselines
across a multi-dataset benchmark, then use ablations to understand which design choices matter.

All runs log to W&B. Use `--no-wandb` to run offline.

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
python benchmark.py --model fcn --suite standard --wandb_group baselines

# ResNet1D — Wang et al. 2017, ~480K params
python benchmark.py --model resnet1d --suite standard --wandb_group baselines

# InceptionTime — Fawaz et al. 2020, ~457K params
python benchmark.py --model inceptiontime --suite standard --wandb_group baselines
```

Expected runtime: ~30–60 min per model on GPU.

---

## 2. Our models — standard suite (apples-to-apples)

Run VNN1D and LaguerreVNN1D with defaults on the same 5 datasets.

```bash
# VNN1D default (quad + cubic symmetric, base_ch=8, ~200K params)
python benchmark.py --model vnn_1d --suite standard --wandb_group our_models

# LaguerreVNN1D default (degrees=[2,3], base_ch=8, ~94K params)
python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group our_models \
  --poly_degrees 2 3
```

---

## 3. VNN1D ablations — quick suite (ECG5000 + FordA only)

Isolate the effect of each architectural choice. Run on the quick 2-dataset suite
to get results fast; promote winners to the standard suite.

```bash
# A1: Quadratic only — no cubic path
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation \
  --disable_cubic

# A2: Quad + cubic symmetric (default)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation

# A3: Quad + cubic general (a·b·c, more expressive, more params)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation \
  --cubic_mode general

# A4: Higher quadratic rank (Q=4 instead of default 2)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation \
  --Q 4

# A5: Larger model width (base_ch=12 to match baseline param counts ~200K→~450K)
python benchmark.py --model vnn_1d --suite quick --wandb_group vnn_ablation \
  --base_ch 12
```

---

## 4. LaguerreVNN1D ablations — quick suite

Explore the polynomial degree and domain scale choices.

```bash
# B1: Degree 2 only — single nonlinear path (quadratic-equivalent)
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2

# B2: Degrees [2, 3] — default, mirrors quad+cubic
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3

# B3: Degrees [2, 3, 4] — three polynomial orders
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 4

# B4: Degrees [2, 3, 4] with tighter input range (alpha=0.5)
#     Recommended when using degree >= 4 to limit polynomial growth
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 4 --alpha 0.5

# B5: Higher orders only — skip degree 2, use [3, 4, 5]
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 3 4 5 --alpha 0.5

# B6: Scaled-up width to match baseline param budget (~450K params at base_ch=16)
python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group laguerre_ablation \
  --poly_degrees 2 3 --base_ch 16
```

---

## 5. Best configs — full 7-dataset suite

Once ablations are done, run the best VNN1D and LaguerreVNN1D configs on the full suite
alongside the top baseline(s) for a definitive table.

```bash
# Best baseline (expected: inceptiontime or resnet1d)
python benchmark.py --model inceptiontime --suite full --wandb_group final

# Best VNN1D config (fill in from ablation results)
python benchmark.py --model vnn_1d --suite full --wandb_group final \
  [--Q ? --base_ch ? --cubic_mode ?]

# Best LaguerreVNN1D config (fill in from ablation results)
python benchmark.py --model laguerre_vnn_1d --suite full --wandb_group final \
  --poly_degrees [? ? ?] --alpha [?] --base_ch [?]
```

---

## Suggested run order

1. **Phase 1+2 in parallel** if you have 2 GPUs: baselines on GPU 0, our models on GPU 1.
2. **Phase 3+4** after Phase 2 finishes (use quick suite, fast ~10 min each).
3. **Phase 5** last, with the best configs identified.

## What to look for

- **Accuracy vs params**: LaguerreVNN1D at ~94K vs baselines at 267–480K.
  If it's within 2–3% accuracy at 4–5× fewer params, that's a compelling result.
- **Which Laguerre degrees help**: B1→B2→B3 progression tells you if higher orders add value.
- **Alpha sensitivity**: compare B3 vs B4 to see if domain mapping matters.
- **VNN vs Laguerre**: compare A2 (quad+cubic) vs B2 (deg=[2,3]) — same param budget,
  different interaction mechanism. This is the cleanest ablation of monomial vs orthogonal.
