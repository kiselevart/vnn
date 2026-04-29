#!/bin/bash
# Phase 7: full-suite S4/S5 + standard-suite S6/S7/S8
#
#   GPU A — S5 (scalar+noclamp)       on full suite      (~145 min)
#   GPU B — S4 (shared+noclamp)       on full suite      (~138 min)
#   GPU C — S6 (learnable α, base)    on standard suite  (~50 min)
#            S7 (shared+scalar+noclamp)on standard suite  (~40 min)
#   GPU D — S8 (scalar+noclamp+learnα)on standard suite  (~50 min)
#
# Phase 5b + Phase 6 results (DONE 2026-04-29, logs/20260429_121626/):
#   S4 std:  95.1%  (shared+noclamp, 18K)
#   S5 std:  95.9%  (scalar+noclamp, 38K) ← standard-suite winner
#   S2 full: 85.7%  (shared, 18K)         ← efficiency headline (= D2 at half params)
#   S3 full: 85.1%  (scalar, 38K)         ← std-suite win doesn't hold on full suite
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

FULL_ARGS=(--suite full     --wandb_group phase7 --no-wandb)
STD_ARGS=( --suite standard --wandb_group phase7 --no-wandb)

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

bench_full() { local name="$1" gpu="$2"; shift 2
  run_job "$name" "$gpu" python benchmark.py "${FULL_ARGS[@]}" "$@"; }

bench_std()  { local name="$1" gpu="$2"; shift 2
  run_job "$name" "$gpu" python benchmark.py "${STD_ARGS[@]}"  "$@"; }

LAG_ARGS=(--poly_degrees 2 3 4 5 --alpha 0.5)

# ---------------------------------------------------------------------------
# GPU A — S5 (scalar+noclamp) on full suite
# ---------------------------------------------------------------------------
gpu_a_jobs() {
  echo "[GPU $GA] S5 scalar+noclamp — full suite ..."
  bench_full "P7_S5_scalar_noclamp_full" $GA --model laguerre_vnn_1d_s5 "${LAG_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# GPU B — S4 (shared+noclamp) on full suite
# ---------------------------------------------------------------------------
gpu_b_jobs() {
  echo "[GPU $GB] S4 shared+noclamp — full suite ..."
  bench_full "P7_S4_shared_noclamp_full" $GB --model laguerre_vnn_1d_s4 "${LAG_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# GPU C — S6 + S7 on standard suite
# ---------------------------------------------------------------------------
gpu_c_jobs() {
  echo "[GPU $GC] S6 learnable-alpha (standard) ..."
  bench_std "P7_S6_learnable_alpha_std" $GC --model laguerre_vnn_1d_s6 "${LAG_ARGS[@]}"
  echo "[GPU $GC] S7 shared+scalar+noclamp (standard) ..."
  bench_std "P7_S7_combo_std"           $GC --model laguerre_vnn_1d_s7 "${LAG_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# GPU D — S8 on standard suite
# ---------------------------------------------------------------------------
gpu_d_jobs() {
  echo "[GPU $GD] S8 scalar+noclamp+learnable-alpha (standard) ..."
  bench_std "P7_S8_scalar_noclamp_la_std" $GD --model laguerre_vnn_1d_s8 "${LAG_ARGS[@]}"
}

# ---------------------------------------------------------------------------
# Launch all streams in parallel
# ---------------------------------------------------------------------------
echo "Phase 7 — 5 jobs across 4 GPUs"
echo "  GPU $GA: S5 scalar+noclamp         (full suite,     ~145 min)"
echo "  GPU $GB: S4 shared+noclamp         (full suite,     ~138 min)"
echo "  GPU $GC: S6 learnable-α + S7 combo (standard suite, ~90 min)"
echo "  GPU $GD: S8 scalar+noclamp+learnα  (standard suite, ~50 min)"
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
[ $SA -eq 0 ] && echo "GPU $GA (S5 full):    done" || echo "GPU $GA (S5 full):    had failures"
[ $SB -eq 0 ] && echo "GPU $GB (S4 full):    done" || echo "GPU $GB (S4 full):    had failures"
[ $SC -eq 0 ] && echo "GPU $GC (S6+S7 std):  done" || echo "GPU $GC (S6+S7 std):  had failures"
[ $SD -eq 0 ] && echo "GPU $GD (S8 std):     done" || echo "GPU $GD (S8 std):     had failures"
echo ""
echo "Logs: $LOG_DIR/"
echo "Next: fill Phase 7 tables in plan.md."
