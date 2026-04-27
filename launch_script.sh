#!/bin/bash
# Phase 4 + Phase 5 (parallel):
#   GPU A — InceptionTime on full suite            (~120 min)
#   GPU B — VNN A1 (no cubic) on full suite        (~100 min)
#   GPU C — Laguerre D2 on full suite              (~130 min)
#   GPU D — Phase 5 simplification ablations ×4    (~130 min)
#            (standard suite, base=D2 config)
#
# Prerequisite: download the 6 extra full-suite datasets before running:
#   python tools/download_ts_datasets.py \
#     --dataset FordB ElectricDevices SpokenArabicDigits \
#               Heartbeat SelfRegulationSCP1 HandMovementDirection
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

LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

FULL_ARGS=(--suite full  --wandb_group phase4 --no-wandb)
STD_ARGS=(--suite standard --wandb_group phase5_simp --no-wandb)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_job() {
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

bench_full() {
  local name="$1" gpu="$2"; shift 2
  run_job "$name" "$gpu" python benchmark.py "${FULL_ARGS[@]}" "$@"
}

bench_std() {
  local name="$1" gpu="$2"; shift 2
  run_job "$name" "$gpu" python benchmark.py "${STD_ARGS[@]}" "$@"
}

# ---------------------------------------------------------------------------
# GPU A — InceptionTime on full suite
# ---------------------------------------------------------------------------
gpu_a_jobs() {
  echo "[GPU $GA] InceptionTime — full suite ..."
  bench_full "IT_full" $GA --model inceptiontime
}

# ---------------------------------------------------------------------------
# GPU B — VNN A1 (no cubic) on full suite
# ---------------------------------------------------------------------------
gpu_b_jobs() {
  echo "[GPU $GB] VNN A1 (no cubic) — full suite ..."
  bench_full "A1_vnn_nocubic_full" $GB --model vnn_1d --disable_cubic
}

# ---------------------------------------------------------------------------
# GPU C — Laguerre D2 on full suite
# ---------------------------------------------------------------------------
gpu_c_jobs() {
  echo "[GPU $GC] Laguerre D2 — full suite ..."
  bench_full "D2_lag_2345_a05_full" $GC --model laguerre_vnn_1d --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# GPU D — Phase 5: simplification ablations on standard suite
#   Base = D2 config ([2,3,4,5] α=0.5)
# ---------------------------------------------------------------------------
gpu_d_jobs() {
  echo "[GPU $GD] Phase 5 simplification ablations — standard suite ..."
  bench_std "P5_base_D2"   $GD --model laguerre_vnn_1d    --poly_degrees 2 3 4 5 --alpha 0.5
  bench_std "P5_S1_noclamp" $GD --model laguerre_vnn_1d_s1 --poly_degrees 2 3 4 5 --alpha 0.5
  bench_std "P5_S2_shared"  $GD --model laguerre_vnn_1d_s2 --poly_degrees 2 3 4 5 --alpha 0.5
  bench_std "P5_S3_scalar"  $GD --model laguerre_vnn_1d_s3 --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# Launch all GPU streams in parallel
# ---------------------------------------------------------------------------
echo "Phase 4 + Phase 5 — 7 jobs across 4 GPUs"
echo "  GPU $GA: InceptionTime (full suite)"
echo "  GPU $GB: VNN A1 no-cubic (full suite)"
echo "  GPU $GC: Laguerre D2 (full suite)"
echo "  GPU $GD: Phase 5 simplification ablations x4 (standard suite)"
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
[ $SA -eq 0 ] && echo "GPU $GA (IT full):       done" || echo "GPU $GA (IT full):       had failures"
[ $SB -eq 0 ] && echo "GPU $GB (A1 full):       done" || echo "GPU $GB (A1 full):       had failures"
[ $SC -eq 0 ] && echo "GPU $GC (D2 full):       done" || echo "GPU $GC (D2 full):       had failures"
[ $SD -eq 0 ] && echo "GPU $GD (Phase 5 simp):  done" || echo "GPU $GD (Phase 5 simp):  had failures"
echo ""
echo "Logs: $LOG_DIR/"
echo "Next: fill Phase 4 result table and Phase 5 simplification table in plan.md."
