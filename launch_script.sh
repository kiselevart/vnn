#!/bin/bash
# Phase 3: Seed runs for the Phase 2.5 winners.
# 4 models × 3 seeds = 12 jobs across 4 GPUs, ~1.5–2 hours wall time.
#
# Winners (8-dataset avg, JV+CT excluded):
#   InceptionTime  460K  95.4%
#   A5: VNN Q=4    78K   95.1%
#   A1: VNN no-cubic 50K 94.9%
#   D2: Laguerre [2,3,4,5] α=0.5  38K  94.7%
#
# GPU layout:
#   GPU A — InceptionTime ×3     (~90 min)
#   GPU B — VNN A1 ×3            (~60 min)
#   GPU C — VNN A5 ×3            (~75 min)
#   GPU D — Laguerre D2 ×3       (~90 min)
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

WANDB_GROUP="phase3_seeds"
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
# GPU A — InceptionTime ×3  (~90 min)
# ---------------------------------------------------------------------------
gpu_a_jobs() {
  echo "[GPU $GA] starting InceptionTime seeds ..."
  bench "IT_s1" $GA --model inceptiontime
  bench "IT_s2" $GA --model inceptiontime
  bench "IT_s3" $GA --model inceptiontime
}

# ---------------------------------------------------------------------------
# GPU B — VNN A1 (no cubic) ×3  (~60 min)
# ---------------------------------------------------------------------------
gpu_b_jobs() {
  echo "[GPU $GB] starting VNN A1 (no cubic) seeds ..."
  bench "A1_vnn_nocubic_s1" $GB --model vnn_1d --disable_cubic
  bench "A1_vnn_nocubic_s2" $GB --model vnn_1d --disable_cubic
  bench "A1_vnn_nocubic_s3" $GB --model vnn_1d --disable_cubic
}

# ---------------------------------------------------------------------------
# GPU C — VNN A5 (Q=4) ×3  (~75 min)
# ---------------------------------------------------------------------------
gpu_c_jobs() {
  echo "[GPU $GC] starting VNN A5 (Q=4) seeds ..."
  bench "A5_vnn_Q4_s1" $GC --model vnn_1d --Q 4
  bench "A5_vnn_Q4_s2" $GC --model vnn_1d --Q 4
  bench "A5_vnn_Q4_s3" $GC --model vnn_1d --Q 4
}

# ---------------------------------------------------------------------------
# GPU D — Laguerre D2 ([2,3,4,5] α=0.5) ×3  (~90 min)
# ---------------------------------------------------------------------------
gpu_d_jobs() {
  echo "[GPU $GD] starting Laguerre D2 seeds ..."
  bench "D2_lag_2345_a05_s1" $GD --model laguerre_vnn_1d --poly_degrees 2 3 4 5 --alpha 0.5
  bench "D2_lag_2345_a05_s2" $GD --model laguerre_vnn_1d --poly_degrees 2 3 4 5 --alpha 0.5
  bench "D2_lag_2345_a05_s3" $GD --model laguerre_vnn_1d --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# Launch all GPU streams in parallel
# ---------------------------------------------------------------------------
echo "Phase 3 — 12 jobs across 4 GPUs (3 seeds each)"
echo "  GPU $GA: InceptionTime ×3"
echo "  GPU $GB: VNN A1 (no cubic) ×3"
echo "  GPU $GC: VNN A5 (Q=4) ×3"
echo "  GPU $GD: Laguerre D2 ([2,3,4,5] α=0.5) ×3"
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
[ $SA -eq 0 ] && echo "GPU $GA (InceptionTime):  all done" || echo "GPU $GA (InceptionTime):  had failures"
[ $SB -eq 0 ] && echo "GPU $GB (VNN A1):         all done" || echo "GPU $GB (VNN A1):         had failures"
[ $SC -eq 0 ] && echo "GPU $GC (VNN A5):         all done" || echo "GPU $GC (VNN A5):         had failures"
[ $SD -eq 0 ] && echo "GPU $GD (Laguerre D2):    all done" || echo "GPU $GD (Laguerre D2):    had failures"
echo ""
echo "Logs: $LOG_DIR/"
echo "Next: compute mean ± std per model from the 3 seeds, update Phase 3 table in plan.md, then proceed to Phase 4."
