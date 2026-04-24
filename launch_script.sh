#!/bin/bash
# Phases 1–4 simultaneous launch across 4 GPUs.
# Edit GPUS to set which physical GPUs to use.

GPUS=(0 1 2 3)   # ← set these to your actual GPU IDs

GA=${GPUS[0]}   # baselines (standard suite)
GB=${GPUS[1]}   # VNN1D standard + ablations
GC=${GPUS[2]}   # LaguerreVNN1D standard + ablations
GD=${GPUS[3]}   # reserved: phase 4.5 / seed runs (not launched here)

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

# ── GPU A: baselines on standard suite (plan section 1) ────────────────────
gpu_a_jobs() {
  local G=baselines
  run_job "1_fcn"           $GA python benchmark.py --model fcn           --suite standard --wandb_group $G --no-wandb
  run_job "1_resnet1d"      $GA python benchmark.py --model resnet1d      --suite standard --wandb_group $G --no-wandb
  run_job "1_inceptiontime" $GA python benchmark.py --model inceptiontime  --suite standard --wandb_group $G --no-wandb
}

# ── GPU B: VNN1D standard then ablations (plan sections 2 + 3) ─────────────
gpu_b_jobs() {
  run_job "2_vnn_standard" $GB python benchmark.py --model vnn_1d --suite standard --wandb_group our_models --no-wandb

  local G=vnn_ablation
  run_job "A1_vnn_no_cubic"      $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb --disable_cubic
  run_job "A2_vnn_default"       $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb
  run_job "A3_vnn_cubic_general" $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb --cubic_mode general
  run_job "A4_vnn_Q1"            $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb --Q 1
  run_job "A5_vnn_Q4"            $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb --Q 4
  run_job "A6_vnn_ch12"          $GB python benchmark.py --model vnn_1d --suite quick --wandb_group $G --no-wandb --base_ch 12
}

# ── GPU C: LaguerreVNN1D standard then ablations (plan sections 2 + 4) ─────
gpu_c_jobs() {
  run_job "2_lag_standard" $GC python benchmark.py --model laguerre_vnn_1d --suite standard --wandb_group our_models --no-wandb --poly_degrees 2 3

  local G=laguerre_ablation
  run_job "B1_lag_deg1"       $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 1
  run_job "B2_lag_deg2"       $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 2
  run_job "B3_lag_deg23"      $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 2 3
  run_job "B4_lag_deg234"     $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 2 3 4
  run_job "B5_lag_deg234_a05" $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 2 3 4 --alpha 0.5
  run_job "B6_lag_deg345_a05" $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 3 4 5 --alpha 0.5
  run_job "B7_lag_deg23_ch16" $GC python benchmark.py --model laguerre_vnn_1d --suite quick --wandb_group $G --no-wandb --poly_degrees 2 3 --base_ch 16
}

# ── Launch ─────────────────────────────────────────────────────────────────
echo "GPU assignment: baselines=$GA  VNN1D=$GB  Laguerre=$GC  free=$GD"
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
[ $SA -eq 0 ] && echo "✓ GPU $GA (baselines) done"       || echo "✗ GPU $GA (baselines) had failures"
[ $SB -eq 0 ] && echo "✓ GPU $GB (VNN1D) done"           || echo "✗ GPU $GB (VNN1D) had failures"
[ $SC -eq 0 ] && echo "✓ GPU $GC (LaguerreVNN1D) done"   || echo "✗ GPU $GC (LaguerreVNN1D) had failures"
echo ""
echo "GPU $GD is free — use it for phase 4.5 (promote winners) and seed runs."
echo "All done. Logs: $LOG_DIR/"
