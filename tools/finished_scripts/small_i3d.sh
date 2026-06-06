#!/bin/bash
# SmallI3D two-stream — UCF101 and HMDB51 3-split means.
#
# width_mult=0.5: all Inception branch channels halved → 6.48M params, 8.05 GFLOPs.
# Parameter-matched to OVN variants (~6.2M) and SmallR3D/SmallR2Plus1D (~5.8M).
#
# Runs 6 splits sequentially (UCF101 splits 1-3, then HMDB51 splits 1-3).
# Estimated runtime: ~0.8h per split → ~5h total on 4 GPUs.
#
# Usage:
#   cd vnn
#   bash small_i3d.sh
#   bash small_i3d.sh 2>&1 | tee logs/small_i3d_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29513

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"           # 4 GPUs × 1e-4
NUM_WORKERS=8
SEED=42
WIDTH_MULT=0.5

WANDB_GROUP="small_i3d_ablation"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/small_i3d_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "SmallI3D two-stream (width_mult=$WIDTH_MULT) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPUs: $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs: $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
echo "  Logs: $LOG_DIR/"
echo ""

# =============================================================================
# Helper
# =============================================================================

run_ddp() {
    local run_name="$1"
    shift
    local log="$LOG_DIR/${run_name}.log"

    {
        echo "=== $run_name"
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT tools/train_i3d_two_stream.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
        tools/train_i3d_two_stream.py "$@" \
        >> "$log" 2>&1
    local status=$?

    {
        echo ""
        echo "================================================================"
        echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
    } >> "$log" 2>&1

    if [ $status -eq 0 ]; then
        local test_acc
        test_acc=$(grep "Top-1:" "$log" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1 || true)
        [ -n "$test_acc" ] && echo "done  (test acc: ${test_acc}%)" || echo "done"
    else
        echo "FAILED (exit $status) — see $log"
    fi

    return $status
}

# =============================================================================
# Runs: UCF101 splits 1-3, then HMDB51 splits 1-3
# =============================================================================

FAILED=()

for DATASET in ucf101 hmdb51; do
    for SPLIT in 1 2 3; do
        RUN_NAME="${DATASET}_small_i3d_seed${SEED}_split${SPLIT}"
        run_ddp "$RUN_NAME" \
            --dataset     "$DATASET" \
            --split       "$SPLIT" \
            --run_name    "$RUN_NAME" \
            --epochs      "$EPOCHS" \
            --batch_size  "$BATCH_SIZE" \
            --lr          "$LR" \
            --num_workers "$NUM_WORKERS" \
            --width_mult  "$WIDTH_MULT" \
            --wandb_group "$WANDB_GROUP" \
            || FAILED+=("$RUN_NAME")
        echo ""
    done
done

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

for DATASET in ucf101 hmdb51; do
    ACCS=()
    for SPLIT in 1 2 3; do
        RUN_NAME="${DATASET}_small_i3d_seed${SEED}_split${SPLIT}"
        LOG="$LOG_DIR/${RUN_NAME}.log"
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            ACC=$(grep "Top-1:" "$LOG" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
            if [ -n "$ACC" ]; then
                echo "  $DATASET split $SPLIT  ✓  ${ACC}%"
                ACCS+=("$ACC")
            else
                echo "  $DATASET split $SPLIT  ✓  (acc not found)"
            fi
        else
            echo "  $DATASET split $SPLIT  ✗  (failed)"
        fi
    done
    if [ ${#ACCS[@]} -eq 3 ]; then
        awk -v a="${ACCS[0]}" -v b="${ACCS[1]}" -v c="${ACCS[2]}" \
            -v ds="$DATASET" 'BEGIN {
            mean = (a + b + c) / 3
            std  = sqrt(((a-mean)^2 + (b-mean)^2 + (c-mean)^2) / 2)
            printf "  %s mean ± std: %.2f ± %.2f%%\n", ds, mean, std
        }'
    fi
    echo ""
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi

echo "Logs: $LOG_DIR/"
