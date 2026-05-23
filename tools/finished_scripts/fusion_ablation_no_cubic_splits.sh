#!/bin/bash
# Fusion ablation (no cubic) — multi-split evaluation.
#
# Runs the same multiplicative vs additive comparison from fusion_ablation_no_cubic.sh
# across additional splits so we can report mean ± std per dataset rather than a
# single split number.
#
# Coverage:
#   UCF101 — splits 2 and 3  (split 1 already done: mult 49.55%, add 51.03%)
#   HMDB51 — splits 1, 2, 3
#
# All runs use: Q=4, float32 (--no_amp), --disable_cubic, seed=42, 4-GPU DDP.
# Runs are sequential (each uses all 4 GPUs).
#
# Usage:
#   cd vnn
#   bash fusion_ablation_no_cubic_splits.sh
#   bash fusion_ablation_no_cubic_splits.sh 2>&1 | tee logs/fusion_ablation_no_cubic_splits_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="4,5,6,7"
NPROC=4

MODELS=("vnn_fusion_ho" "vnn_additive_fusion_ho")

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
EXTRA_ARGS="--no_amp --disable_cubic"

SEED=42
MASTER_PORT=29500
WANDB_GROUP="fusion_ablation_no_cubic_splits"

# Splits to run per dataset (split 1 UCF101 already done separately)
declare -A DATASET_SPLITS
DATASET_SPLITS["ucf101"]="2 3"
DATASET_SPLITS["hmdb51"]="1 2 3"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/fusion_ablation_no_cubic_splits_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Fusion ablation (no cubic, multi-split) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Models:   ${MODELS[*]}"
echo "  UCF101 splits: ${DATASET_SPLITS[ucf101]}"
echo "  HMDB51 splits: ${DATASET_SPLITS[hmdb51]}"
echo "  Cubic:    DISABLED (Q auto-boosted)"
echo "  Seed:     $SEED"
echo "  GPUs:     $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
echo "  Logs:     $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/<run>.log"
echo "  grep -h 'Test Result\|exit=' $LOG_DIR/*.log"
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
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT train_par.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_par.py "$@" \
        >> "$log" 2>&1
    local status=$?

    {
        echo ""
        echo "================================================================"
        echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
    } >> "$log" 2>&1

    if [ $status -eq 0 ]; then
        local test_acc
        test_acc=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$log" 2>/dev/null | tail -1 || true)
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

for DATASET in ucf101 hmdb51; do
    read -ra SPLITS <<< "${DATASET_SPLITS[$DATASET]}"
    echo "=== Dataset: $DATASET (splits: ${SPLITS[*]}) ==="
    for MODEL in "${MODELS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_no_cubic_split${SPLIT}"
            run_ddp "$RUN_NAME" \
                --dataset     "$DATASET" \
                --model       "$MODEL" \
                --split       "$SPLIT" \
                --epochs      "$EPOCHS" \
                --batch_size  "$BATCH_SIZE" \
                --lr          "$LR" \
                --num_workers "$NUM_WORKERS" \
                --wandb_group "$WANDB_GROUP" \
                --run_name    "$RUN_NAME" \
                --seed        "$SEED" \
                $EXTRA_ARGS \
                || FAILED+=("$RUN_NAME")
        done
    done
    echo ""
done

# =============================================================================
# Summary — per-model averages across splits
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""
echo "Results by dataset / model / split:"
echo ""

# UCF101: include split 1 result from the previous run (hardcoded)
echo "  UCF101:"
echo "    (split 1 from fusion_ablation_no_cubic_20260522_110905 — not rerun here)"
echo "    vnn_fusion_ho           split 1: 49.55%"
echo "    vnn_additive_fusion_ho  split 1: 51.03%"
echo ""

for DATASET in ucf101 hmdb51; do
    read -ra SPLITS <<< "${DATASET_SPLITS[$DATASET]}"
    echo "  $DATASET (new splits):"
    for MODEL in "${MODELS[@]}"; do
        echo "    $MODEL:"
        local_sum=0
        local_count=0
        for SPLIT in "${SPLITS[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_no_cubic_split${SPLIT}"
            LOG="$LOG_DIR/${RUN_NAME}.log"
            if grep -q "exit=0" "$LOG" 2>/dev/null; then
                ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || echo "")
                if [ -n "$ACC" ]; then
                    echo "      split $SPLIT  ✓  test acc: ${ACC}%"
                    local_sum=$(awk "BEGIN {printf \"%.2f\", $local_sum + $ACC}")
                    local_count=$((local_count + 1))
                else
                    echo "      split $SPLIT  ✓  (acc not found in log)"
                fi
            else
                echo "      split $SPLIT  ✗  (failed — check $LOG)"
            fi
        done
        if [ "$local_count" -gt 0 ]; then
            AVG=$(awk "BEGIN {printf \"%.2f\", $local_sum / $local_count}")
            echo "      avg (splits ${SPLITS[*]}): ${AVG}%"
        fi
    done
    echo ""
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
echo ""
echo "To compute UCF101 3-split average including split 1:"
echo "  mult:  avg(49.55, split2, split3)"
echo "  add:   avg(51.03, split2, split3)"
