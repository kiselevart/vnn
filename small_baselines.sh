#!/bin/bash
# Parameter-matched baselines for the LVN/ortho comparison.
#
# Models (~6.2M params — budget-matched to LVN/ortho fusion models):
#   SmallR3D      — R3D-18 style, channels [32,64,128,192], pure 3D Conv
#   SmallR2Plus1D — R(2+1)D-18 style, same channels, factorized spatiotemporal
#
# Runs 3 splits × 2 datasets (UCF101 + HMDB51) per model = 12 runs total.
# Reports per-split and mean ± std — matches LVN/ortho reporting protocol.
#
# Usage:
#   cd vnn && bash small_baselines.sh
#   cd vnn && bash small_baselines.sh 2>&1 | tee logs/small_baselines_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

DDP_GPUS="0,1,2,3"
DDP_NPROC=4
DDP_MASTER_PORT=29506   # ports 29500-29505 reserved for other experiments

SEED=42
EPOCHS=100

# DDP training params
DDP_BATCH_SIZE=8        # per GPU
DDP_LR="4e-4"           # 4 GPUs × 1e-4
DDP_NUM_WORKERS=4

WANDB_GROUP="small_baselines"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/small_baselines_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Small baselines — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Models:  SmallR3D, SmallR2Plus1D  (~6.2M params)"
echo "  Splits:  UCF101 1-3, HMDB51 1-3"
echo "  Seed:    $SEED"
echo "  Epochs:  $EPOCHS"
echo "  DDP GPUs:  $DDP_GPUS  (nproc=$DDP_NPROC)  port=$DDP_MASTER_PORT"
echo "  Logs:    $LOG_DIR/"
echo ""

# Accumulate run names for the summary
ALL_RUNS=()

# =============================================================================
# Helper: DDP run
# =============================================================================

run_ddp() {
    local run_name="$1"
    shift
    local log="$LOG_DIR/${run_name}.log"
    ALL_RUNS+=("$run_name")

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
# SmallR3D — UCF101 splits 1-3, HMDB51 splits 1-3
# =============================================================================

echo "--- SmallR3D ---"
for DATASET in ucf101 hmdb51; do
    for SPLIT in 1 2 3; do
        RUN="${DATASET}_small_r3d_seed${SEED}_split${SPLIT}"
        run_ddp "$RUN" \
            --dataset     "$DATASET" \
            --model       small_r3d \
            --split       "$SPLIT" \
            --epochs      "$EPOCHS" \
            --batch_size  "$DDP_BATCH_SIZE" \
            --lr          "$DDP_LR" \
            --num_workers "$DDP_NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP" \
            --run_name    "$RUN" \
            --seed        "$SEED"
    done
done
echo ""

# =============================================================================
# SmallR2Plus1D — UCF101 splits 1-3, HMDB51 splits 1-3
# =============================================================================

echo "--- SmallR2Plus1D ---"
for DATASET in ucf101 hmdb51; do
    for SPLIT in 1 2 3; do
        RUN="${DATASET}_small_r2plus1d_seed${SEED}_split${SPLIT}"
        run_ddp "$RUN" \
            --dataset     "$DATASET" \
            --model       small_r2plus1d \
            --split       "$SPLIT" \
            --epochs      "$EPOCHS" \
            --batch_size  "$DDP_BATCH_SIZE" \
            --lr          "$DDP_LR" \
            --num_workers "$DDP_NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP" \
            --run_name    "$RUN" \
            --seed        "$SEED"
    done
done
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

extract_test_acc() {
    grep -oP "(?<=Test acc: )\S+" "$1" 2>/dev/null | tail -1 || echo "?"
}

print_model_stats() {
    local model_label="$1"
    local model_key="$2"

    echo "  $model_label"
    for DATASET in ucf101 hmdb51; do
        local accs=()
        for SPLIT in 1 2 3; do
            local rname="${DATASET}_${model_key}_seed${SEED}_split${SPLIT}"
            local log="$LOG_DIR/${rname}.log"
            if grep -q "exit=0" "$log" 2>/dev/null; then
                accs+=("$(extract_test_acc "$log")")
            else
                accs+=("FAIL")
            fi
        done
        echo "    $DATASET: split1=${accs[0]}  split2=${accs[1]}  split3=${accs[2]}"
    done
    echo ""
}

print_model_stats "SmallR3D" "small_r3d"
print_model_stats "SmallR2Plus1D" "small_r2plus1d"

echo "Logs: $LOG_DIR/"
echo "Next: compute mean ± std and add to Section 6 comparison table."
