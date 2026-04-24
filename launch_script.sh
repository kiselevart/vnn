#!/bin/bash
# Phase 2: Fix + promote winners — see plan.md for context.
# Edit GPUS to set your physical GPU IDs.

GPUS=(0 1 2 3)   # ← set these to your actual GPU IDs

GA=${GPUS[0]}   # Ethanol reruns (all models, 800 epochs)
GB=${GPUS[1]}   # A3 rerun + VNN1D winner candidates on standard
GC=${GPUS[2]}   # Laguerre winner candidates on standard
GD=${GPUS[3]}   # free — seed runs once phase 2 winners are known

LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_job() {
  local name="$1" gpu="$2"
  shift 2
  local log="$LOG_DIR/${name}.log"
  local full_cmd="CUDA_VISIBLE_DEVICES=$gpu $*"

  {
    echo "=== $name"
    echo "CMD: $full_cmd"
    echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
  } > "$log" 2>&1

  CUDA_VISIBLE_DEVICES="$gpu" "$@" >> "$log" 2>&1
  local status=$?

  {
    echo ""
    echo "================================================================"
    echo "CMD:   $full_cmd"
    echo "END:   $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
  } >> "$log" 2>&1

  [ $status -eq 0 ] && echo "✓ $name" || echo "✗ $name (exit $status)"
  return $status
}

# ── GPU A: Ethanol reruns with 800 epochs (plan section 2a) ────────────────
gpu_a_jobs() {
  local E="--datasets EthanolConcentration --epochs 800 --no-wandb"
  run_job "eth_fcn"          $GA python benchmark.py --model fcn           $E
  run_job "eth_resnet1d"     $GA python benchmark.py --model resnet1d      $E
  run_job "eth_inceptiontime" $GA python benchmark.py --model inceptiontime $E
  run_job "eth_vnn_Q1_nocubic" $GA python benchmark.py --model vnn_1d $E --Q 1 --disable_cubic
  run_job "eth_lag_B6"       $GA python benchmark.py --model laguerre_vnn_1d $E --poly_degrees 3 4 5 --alpha 0.5
}

# ── GPU B: A3 rerun + VNN1D winners on standard (plan sections 2b + 2c) ────
gpu_b_jobs() {
  # A3: was broken (--cubic_mode not parsed by benchmark.py) — now fixed
  run_job "A3_vnn_cubic_general" $GB python benchmark.py --model vnn_1d --suite quick \
    --wandb_group vnn_ablation --no-wandb --cubic_mode general

  # C1: Q=1, cubic symmetric — ablation FordA winner
  run_job "C1_vnn_Q1_standard" $GB python benchmark.py --model vnn_1d --suite standard \
    --wandb_group vnn_winners --no-wandb --Q 1

  # C2: Q=1, no cubic — most minimal config; does dropping cubic hurt on harder datasets?
  run_job "C2_vnn_Q1_nocubic_standard" $GB python benchmark.py --model vnn_1d --suite standard \
    --wandb_group vnn_winners --no-wandb --Q 1 --disable_cubic
}

# ── GPU C: Laguerre winners on standard (plan section 2d) ──────────────────
gpu_c_jobs() {
  # C3: deg=1 — linear Laguerre, best ECG5000, fewest params (17K)
  run_job "C3_lag_deg1_standard" $GC python benchmark.py --model laguerre_vnn_1d --suite standard \
    --wandb_group lag_winners --no-wandb --poly_degrees 1

  # C4: deg=[3,4,5] α=0.5 — best on FordA, may handle harder datasets better
  run_job "C4_lag_B6_standard" $GC python benchmark.py --model laguerre_vnn_1d --suite standard \
    --wandb_group lag_winners --no-wandb --poly_degrees 3 4 5 --alpha 0.5
}

# ── Launch ─────────────────────────────────────────────────────────────────
echo "GPU assignment: Ethanol=$GA  VNN=$GB  Laguerre=$GC  free=$GD"
echo "Logs: $LOG_DIR/"
echo ""
echo "To monitor:"
echo "  tail -f $LOG_DIR/*.log"
echo "  watch -n5 'ls -lh $LOG_DIR/'"
echo ""

gpu_a_jobs &
PID_A=$!
gpu_b_jobs &
PID_B=$!
gpu_c_jobs &
PID_C=$!

wait $PID_A; SA=$?
wait $PID_B; SB=$?
wait $PID_C; SC=$?

echo ""
[ $SA -eq 0 ] && echo "✓ GPU $GA (Ethanol) done"      || echo "✗ GPU $GA (Ethanol) had failures"
[ $SB -eq 0 ] && echo "✓ GPU $GB (VNN1D) done"        || echo "✗ GPU $GB (VNN1D) had failures"
[ $SC -eq 0 ] && echo "✓ GPU $GC (Laguerre) done"     || echo "✗ GPU $GC (Laguerre) had failures"
echo ""
echo "GPU $GD is free — use it for seed runs once you've picked the phase 2 winners."
echo "All done. Logs: $LOG_DIR/"
