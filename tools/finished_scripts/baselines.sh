#!/bin/bash
# Baseline comparison for the VNN paper.
#
# Models (run sequentially, all UCF101 split 1):
#   R3D-18          — pure 3D ResNet-18 (Conv3D, no factorization)
#   R(2+1)D-18      — factorized spatiotemporal conv (space then time)
#   ResNet50FrameAvg— 2D ResNet-50, per-frame + temporal average: no motion, spatial-only ceiling
#
# All models run via train_par.py (4-GPU DDP).
#
# Assumes GPUs 4,5,6,7 are busy with fusion_ablation.sh.
#
# Usage:
#   bash baselines.sh
#   bash baselines.sh 2>&1 | tee logs/baselines_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

DDP_GPUS="0,1,2,3"
DDP_NPROC=4
DDP_MASTER_PORT=29501   # different from fusion_ablation.sh (29500)

DATASET="ucf101"
SEED=42
EPOCHS=100
SPLIT=1

# DDP training params — match VNN ablation conditions
DDP_BATCH_SIZE=8        # per GPU
DDP_LR="4e-4"           # 4 GPUs × 1e-4
DDP_NUM_WORKERS=4

WANDB_GROUP="baselines"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/baselines_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Baselines — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:     $DATASET  (split $SPLIT)"
echo "  Seed:        $SEED"
echo "  Epochs:      $EPOCHS"
echo "  DDP GPUs:    $DDP_GPUS  (nproc=$DDP_NPROC)  port=$DDP_MASTER_PORT"
echo "  Logs:        $LOG_DIR/"
echo ""

# =============================================================================
# Helper: DDP run via train_par.py
# =============================================================================

run_ddp() {
    local run_name="$1"
    shift
    local log="$LOG_DIR/${run_name}.log"

    {
        echo "=== $run_name"
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$DDP_GPUS torchrun --nproc_per_node=$DDP_NPROC --master_port=$DDP_MASTER_PORT train_par.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$DDP_GPUS" \
        torchrun --nproc_per_node="$DDP_NPROC" --master_port="$DDP_MASTER_PORT" train_par.py "$@" \
        >> "$log" 2>&1
    local status=$?

    {
        echo ""
        echo "================================================================"
        echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
    } >> "$log" 2>&1

    if [ $status -eq 0 ]; then
        local best
        best=$(grep -oP "(?<=Best acc: )\S+" "$log" 2>/dev/null | tail -1 || true)
        [ -n "$best" ] && echo "done  (best val acc: $best)" || echo "done"
    else
        echo "FAILED (exit $status) — see $log"
    fi

    return $status
}

# =============================================================================
# R3D-18  (pure Conv3D)
# =============================================================================

echo "--- R3D-18 (Conv3D) ---"
R3D_RUN="ucf101_r3d_seed${SEED}_baseline"
run_ddp "$R3D_RUN" \
    --dataset     "$DATASET" \
    --model       r3d \
    --split       "$SPLIT" \
    --epochs      "$EPOCHS" \
    --batch_size  "$DDP_BATCH_SIZE" \
    --lr          "$DDP_LR" \
    --num_workers "$DDP_NUM_WORKERS" \
    --wandb_group "$WANDB_GROUP" \
    --run_name    "$R3D_RUN" \
    --seed        "$SEED"
echo ""

# =============================================================================
# R(2+1)D-18  (factorized spatiotemporal conv)
# =============================================================================

echo "--- R(2+1)D-18 ---"
R2P1D_RUN="ucf101_r2plus1d_seed${SEED}_baseline"
run_ddp "$R2P1D_RUN" \
    --dataset     "$DATASET" \
    --model       r2plus1d \
    --split       "$SPLIT" \
    --epochs      "$EPOCHS" \
    --batch_size  "$DDP_BATCH_SIZE" \
    --lr          "$DDP_LR" \
    --num_workers "$DDP_NUM_WORKERS" \
    --wandb_group "$WANDB_GROUP" \
    --run_name    "$R2P1D_RUN" \
    --seed        "$SEED"
echo ""

# =============================================================================
# ResNet50 frame-average  (2D spatial only, no temporal modeling)
# =============================================================================

echo "--- ResNet-50 frame-avg (2D spatial baseline) ---"
RN50_RUN="ucf101_resnet50_frame_avg_seed${SEED}_baseline"
run_ddp "$RN50_RUN" \
    --dataset     "$DATASET" \
    --model       resnet50_frame_avg \
    --split       "$SPLIT" \
    --epochs      "$EPOCHS" \
    --batch_size  "$DDP_BATCH_SIZE" \
    --lr          "$DDP_LR" \
    --num_workers "$DDP_NUM_WORKERS" \
    --wandb_group "$WANDB_GROUP" \
    --run_name    "$RN50_RUN" \
    --seed        "$SEED"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

for RNAME in "$R3D_RUN" "$R2P1D_RUN" "$RN50_RUN"; do
    LOG="$LOG_DIR/${RNAME}.log"
    if grep -q "exit=0" "$LOG" 2>/dev/null; then
        BEST=$(grep -oP "(?<=Best acc: )\S+" "$LOG" 2>/dev/null | tail -1 || echo "?")
        echo "  $RNAME  ✓  best val acc: $BEST"
    else
        echo "  $RNAME  ✗  (failed — check $LOG)"
    fi
done

echo ""
echo "Logs: $LOG_DIR/"
echo "Next: add test acc to baselines table in paper Section 6."
