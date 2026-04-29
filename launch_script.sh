#!/bin/bash
# Phase 5b + Phase 6 (parallel):
#   GPU A — Laguerre S3 (scalar gates) on full suite        (~130 min)
#   GPU B — Laguerre S2 (shared proj)  on full suite        (~130 min)
#   GPU D — S4 (shared+noclamp) + S5 (scalar+noclamp)       (~100 min)
#            on standard suite
#
# Phase 4 + Phase 5 (DONE 2026-04-28, logs/20260428_131623/):
#   IT full: 87.1% avg (15 datasets)
#   A1 full: 85.5% avg — VNN no-cubic
#   D2 full: 85.6% avg — Laguerre [2,3,4,5] α=0.5
#   Base (D2) std: 95.3%, S1: 95.1%, S2: 95.4% (18K), S3: 95.8%
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

FULL_ARGS=(--suite full  --wandb_group phase6 --no-wandb)
STD_ARGS=(--suite standard --wandb_group phase6_std --no-wandb)

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
# GPU A — Laguerre S3 (scalar gates) on full suite
# ---------------------------------------------------------------------------
gpu_a_jobs() {
  echo "[GPU $GA] Laguerre S3 (scalar gates) — full suite ..."
  bench_full "P6_S3_scalar_full" $GA --model laguerre_vnn_1d_s3 --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# GPU B — Laguerre S2 (shared proj) on full suite
# ---------------------------------------------------------------------------
gpu_b_jobs() {
  echo "[GPU $GB] Laguerre S2 (shared proj) — full suite ..."
  bench_full "P6_S2_shared_full" $GB --model laguerre_vnn_1d_s2 --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# GPU D — Phase 5b: S4 + S5 ablations on standard suite
# ---------------------------------------------------------------------------
gpu_d_jobs() {
  echo "[GPU $GD] Phase 5b: S4 (shared+noclamp) + S5 (scalar+noclamp) — standard suite ..."
  bench_std "P5_S4_shared_noclamp" $GD --model laguerre_vnn_1d_s4 --poly_degrees 2 3 4 5 --alpha 0.5
  bench_std "P5_S5_scalar_noclamp" $GD --model laguerre_vnn_1d_s5 --poly_degrees 2 3 4 5 --alpha 0.5
}

# ---------------------------------------------------------------------------
# Launch all streams in parallel
# ---------------------------------------------------------------------------
echo "Phase 5b + Phase 6 — 4 jobs across 3 GPUs"
echo "  GPU $GA: Laguerre S3 scalar gates (full suite)"
echo "  GPU $GB: Laguerre S2 shared proj  (full suite)"
echo "  GPU $GD: S4 shared+noclamp + S5 scalar+noclamp (standard suite)"
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
gpu_d_jobs &
PID_D=$!

wait $PID_A; SA=$?
wait $PID_B; SB=$?
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
[ $SA -eq 0 ] && echo "GPU $GA (S3 full):    done" || echo "GPU $GA (S3 full):    had failures"
[ $SB -eq 0 ] && echo "GPU $GB (S2 full):    done" || echo "GPU $GB (S2 full):    had failures"
[ $SD -eq 0 ] && echo "GPU $GD (S4+S5 std):  done" || echo "GPU $GD (S4+S5 std):  had failures"
echo ""
echo "Logs: $LOG_DIR/"
echo "Next: fill Phase 5 S4/S5 rows and Phase 6 table in plan.md."
