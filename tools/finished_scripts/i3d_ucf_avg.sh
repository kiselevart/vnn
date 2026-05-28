#!/bin/bash
# Two-stream I3D — UCF101 3-split mean for the paper's Section 6.3 baseline.
#
# Runs splits 1, 2, 3 sequentially (each uses all 4 GPUs via DDP).
# Reports per-split test acc and mean ± std at the end.
#
# Estimated runtime: ~2.5h per split → ~7.5h total.
#
# Usage:
#   cd vnn
#   bash i3d_ucf_avg.sh
#   bash i3d_ucf_avg.sh 2>&1 | tee logs/i3d_ucf_avg_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29502   # reserved for I3D

DATASET="ucf101"
SPLITS=(1 2 3)

EPOCHS=100
BATCH_SIZE=8        # 16 (default) risks OOM; 8 matches VNN ablation conditions
LR="4e-4"           # 4 GPUs × 1e-4
NUM_WORKERS=8
SEED=42

WANDB_GROUP="i3d_twostream_ucf101"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/i3d_ucf_avg_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "I3D two-stream — UCF101 3-split avg — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Splits:   ${SPLITS[*]}"
echo "  GPUs:     $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
echo "  Logs:     $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/<run>.log"
echo "  grep -h 'Top-1\|exit=' $LOG_DIR/*.log"
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
# Runs
# =============================================================================

FAILED=()

for SPLIT in "${SPLITS[@]}"; do
    RUN_NAME="ucf101_i3d_twostream_seed${SEED}_split${SPLIT}"
    run_ddp "$RUN_NAME" \
        --dataset     "$DATASET" \
        --split       "$SPLIT" \
        --run_name    "$RUN_NAME" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --wandb_group "$WANDB_GROUP" \
        || FAILED+=("$RUN_NAME")
    echo ""
done

# =============================================================================
# Summary — per-split results and mean ± std
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

ACCS=()
for SPLIT in "${SPLITS[@]}"; do
    RUN_NAME="ucf101_i3d_twostream_seed${SEED}_split${SPLIT}"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    if grep -q "exit=0" "$LOG" 2>/dev/null; then
        ACC=$(grep "Top-1:" "$LOG" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
        if [ -n "$ACC" ]; then
            echo "  split $SPLIT  ✓  test acc: ${ACC}%"
            ACCS+=("$ACC")
        else
            echo "  split $SPLIT  ✓  (acc not found in log)"
        fi
    else
        echo "  split $SPLIT  ✗  (failed — check $LOG)"
    fi
done

echo ""

# Compute mean and std if all 3 splits succeeded
if [ ${#ACCS[@]} -eq 3 ]; then
    awk -v a="${ACCS[0]}" -v b="${ACCS[1]}" -v c="${ACCS[2]}" 'BEGIN {
        mean = (a + b + c) / 3
        std  = sqrt(((a-mean)^2 + (b-mean)^2 + (c-mean)^2) / 2)
        printf "  Mean ± std: %.2f ± %.2f%%\n", mean, std
    }'
elif [ ${#ACCS[@]} -gt 0 ]; then
    awk -v n="${#ACCS[@]}" -v a="${ACCS[0]}" -v b="${ACCS[1]:-0}" 'BEGIN {
        printf "  Partial mean (%d splits): %.2f%%\n", n, (a + b) / n
    }'
fi

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
echo "Next: add 3-split mean ± std to Section 6.3 alongside R3D / R(2+1)D."
