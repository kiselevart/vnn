#!/bin/bash
# Phase 2.5: Full ablation rerun on the expanded 10-dataset standard suite.
# 18 jobs total — 4 GPUs running in parallel, ~2–3 hours wall time.
#
# GPU layout (edit GPUS to match your actual device IDs):
#   GPU A — baselines (FCN / ResNet1D / InceptionTime) + VNN A6     (~140 min)
#   GPU B — VNN ablations A1–A4                                      (~90 min)
#   GPU C — VNN A5 + Laguerre B1–B4                                  (~110 min)
#   GPU D — Laguerre B5–B6, D1–D3                                    (~125 min)
#
# Usage:
#   bash launch_script.sh
#   # Logs land in ./logs/<timestamp>/  — one file per job.

set -euo pipefail

GPUS=(0 1 6 7)
GA=${GPUS[0]}
GB=${GPUS[1]}
GC=${GPUS[2]}
GD=${GPUS[3]}

WANDB_GROUP="phase25"
SUITE_ARGS=(--suite standard --wandb_group "$WANDB_GROUP" --no-wandb)

LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_job() {
  # run_job <name> <gpu> <cmd...>
  local name="$1" gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"

  {
    echo "=== $name"
    echo "CMD: CUDA_VISIBLE_DEVICES=$gpu $*"
    echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
  } > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$log" 2>&1
  local status=$?

  {
    echo ""
    echo "================================================================"
    echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
  } >> "$log" 2>&1

  [ $status -eq 0 ] && echo "  ✓ $name" || echo "  ✗ $name (exit $status)"
  return $status
}

bench() {
  # bench <name> <gpu> [extra benchmark.py args...]
  local name="$1" gpu="$2"
  shift 2
  run_job "$name" "$gpu" python benchmark.py "${SUITE_ARGS[@]}" "$@"
}

# ---------------------------------------------------------------------------
# Pre-flight: download all standard-suite datasets on the main process
# so parallel GPU jobs don't race to write the same files.
# ---------------------------------------------------------------------------

echo "Downloading standard-suite datasets (skips already-present files) ..."
python tools/download_ts_datasets.py \
  --dataset ECG5000 FordA Wafer \
            ArticularyWordRecognition NATOPS JapaneseVowels \
            Epilepsy BasicMotions CharacterTrajectories UWaveGestureLibrary
echo ""

# ---------------------------------------------------------------------------
# GPU A — Baselines + VNN A6  (4 jobs, ~140 min)
# Large models (265–479K params) are slowest, so kept together on one GPU.
# ---------------------------------------------------------------------------
gpu_a_jobs() {
  echo "[GPU $GA] starting baselines + A6 ..."
  bench "fcn"           $GA --model fcn
  bench "resnet1d"      $GA --model resnet1d
  bench "inceptiontime" $GA --model inceptiontime
  bench "A6_vnn_ch12"   $GA --model vnn_1d --base_ch 12
}

# ---------------------------------------------------------------------------
# GPU B — VNN1D ablations A1–A4  (4 jobs, ~90 min)
# ---------------------------------------------------------------------------
gpu_b_jobs() {
  echo "[GPU $GB] starting VNN ablations A1–A4 ..."
  bench "A1_vnn_nocubic"  $GB --model vnn_1d --disable_cubic
  bench "A2_vnn_default"  $GB --model vnn_1d
  bench "A3_vnn_cubicgen" $GB --model vnn_1d --cubic_mode general
  bench "A4_vnn_Q1"       $GB --model vnn_1d --Q 1
}

# ---------------------------------------------------------------------------
# GPU C — VNN A5 + Laguerre B1–B4  (5 jobs, ~110 min)
# ---------------------------------------------------------------------------
gpu_c_jobs() {
  echo "[GPU $GC] starting VNN A5 + Laguerre B1–B4 ..."
  bench "A5_vnn_Q4"    $GC --model vnn_1d --Q 4
  bench "B1_lag_1"     $GC --model laguerre_vnn_1d --poly_degrees 1 --alpha 1.0
  bench "B2_lag_2"     $GC --model laguerre_vnn_1d --poly_degrees 2 --alpha 1.0
  bench "B3_lag_23"    $GC --model laguerre_vnn_1d --poly_degrees 2 3 --alpha 1.0
  bench "B4_lag_234"   $GC --model laguerre_vnn_1d --poly_degrees 2 3 4 --alpha 1.0
}

# ---------------------------------------------------------------------------
# GPU D — Laguerre B5–B6, D1–D3  (5 jobs, ~125 min)
# ---------------------------------------------------------------------------
gpu_d_jobs() {
  echo "[GPU $GD] starting Laguerre B5–B6 + D1–D3 ..."
  bench "B5_lag_234_a05"  $GD --model laguerre_vnn_1d --poly_degrees 2 3 4   --alpha 0.5
  bench "B6_lag_345_a05"  $GD --model laguerre_vnn_1d --poly_degrees 3 4 5   --alpha 0.5
  bench "D1_lag_34_a05"   $GD --model laguerre_vnn_1d --poly_degrees 3 4     --alpha 0.5
  bench "D2_lag_2345_a05" $GD --model laguerre_vnn_1d --poly_degrees 2 3 4 5 --alpha 0.5
  bench "D3_lag_456_a05"  $GD --model laguerre_vnn_1d --poly_degrees 4 5 6   --alpha 0.5
}

# ---------------------------------------------------------------------------
# Launch all GPU streams in parallel
# ---------------------------------------------------------------------------
echo "Phase 2.5 — 18 jobs across 4 GPUs"
echo "  GPU $GA: baselines (FCN / ResNet1D / InceptionTime) + VNN A6"
echo "  GPU $GB: VNN A1–A4"
echo "  GPU $GC: VNN A5 + Laguerre B1–B4"
echo "  GPU $GD: Laguerre B5–B6, D1–D3"
echo "W&B group: $WANDB_GROUP"
echo "Logs: $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  watch -n10 'grep -rh \"✓\|✗\" $LOG_DIR/ 2>/dev/null | sort'"
echo ""

gpu_a_jobs &
PID_A=$!
gpu_b_jobs &
PID_B=$!
gpu_c_jobs &
PID_C=$!
gpu_d_jobs &
PID_D=$!

wait $PID_A; SA=$?
wait $PID_B; SB=$?
wait $PID_C; SC=$?
wait $PID_D; SD=$?

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================== SUMMARY ==============================="
for log in "$LOG_DIR"/*.log; do
  name=$(basename "$log" .log)
  if grep -q "exit=0" "$log" 2>/dev/null; then
    echo "  ✓ $name"
  else
    echo "  ✗ $name"
  fi
done | sort

echo ""
[ $SA -eq 0 ] && echo "GPU $GA (baselines+A6): all done" || echo "GPU $GA (baselines+A6): had failures"
[ $SB -eq 0 ] && echo "GPU $GB (VNN A1-A4):    all done" || echo "GPU $GB (VNN A1-A4):    had failures"
[ $SC -eq 0 ] && echo "GPU $GC (VNN A5+Lag B): all done" || echo "GPU $GC (VNN A5+Lag B): had failures"
[ $SD -eq 0 ] && echo "GPU $GD (Lag B5-D3):    all done" || echo "GPU $GD (Lag B5-D3):    had failures"
echo ""
echo "Logs: $LOG_DIR/"
echo "Next: fill Phase 2.5 result tables in plan.md, then pick winners for Phase 3."
