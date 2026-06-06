#!/bin/bash
# Q sweep rerun — vnn_additive_fusion_ho across all 3 UCF101 splits, Q=16 down to Q=1.
#
# Sweeps backbone quadratic rank Q on the modern additive HO model (AMP-safe, gated,
# clamped — no --no_amp needed). Fusion head Q is fixed at its default (Q=2); this
# sweep isolates backbone capacity per Proposition 1.
#
# Runs Q values in descending order so high-Q results arrive first.
# Outer loop: Q (16,8,4,2,1); inner loop: splits (1,2,3).
#
# Config: vnn_additive_fusion_ho, AMP, seed=42, 4-GPU DDP.
# Estimated runtime: ~7h on 4 GPUs.
#
# Usage:
#   cd vnn
#   bash q_sweep_rerun.sh
#   bash q_sweep_rerun.sh 2>&1 | tee logs/q_sweep_rerun_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29520

DATASET="ucf101"
MODEL="vnn_additive_fusion_ho"
Q_VALUES=(16 8 4 2 1)
SPLITS=(1 2 3)

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
EXTRA_ARGS=""

SEED=42
WANDB_GROUP="q_sweep_additive"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/q_sweep_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TOTAL=$(( ${#Q_VALUES[@]} * ${#SPLITS[@]} ))

echo "Q sweep rerun — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:  $DATASET"
echo "  Model:    $MODEL"
echo "  Q values: ${Q_VALUES[*]}  (descending)"
echo "  Splits:   ${SPLITS[*]}"
echo "  Total:    $TOTAL runs"
echo "  Seed:     $SEED"
echo "  GPUs:     $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
echo "  W&B group: $WANDB_GROUP"
echo "  Logs:     $LOG_DIR/"
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
# Sweep — outer Q (descending), inner splits
# =============================================================================

FAILED=()
RUN_NUM=0

for Q in "${Q_VALUES[@]}"; do
    echo "--- Q=$Q ---"
    for SPLIT in "${SPLITS[@]}"; do
        RUN_NUM=$(( RUN_NUM + 1 ))
        RUN_NAME="${DATASET}_${MODEL}_Q${Q}_split${SPLIT}_rerun"
        echo -n "  [$RUN_NUM/$TOTAL] "
        run_ddp "$RUN_NAME" \
            --dataset     "$DATASET" \
            --model       "$MODEL" \
            --Q           "$Q" \
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
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

printf "  %-6s  %-10s  %-10s  %-10s  %-14s\n" "Q" "split 1" "split 2" "split 3" "mean ± std"
printf "  %-6s  %-10s  %-10s  %-10s  %-14s\n" "------" "----------" "----------" "----------" "--------------"

for Q in "${Q_VALUES[@]}"; do
    ACCS=()
    for SPLIT in "${SPLITS[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_Q${Q}_split${SPLIT}_rerun"
        LOG="$LOG_DIR/${RUN_NAME}.log"
        ACC=""
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || echo "")
        fi
        ACCS+=("${ACC:-FAILED}")
    done

    MEAN_STD=""
    if [[ "${ACCS[0]}" =~ ^[0-9]+\.[0-9]+$ ]] && \
       [[ "${ACCS[1]}" =~ ^[0-9]+\.[0-9]+$ ]] && \
       [[ "${ACCS[2]}" =~ ^[0-9]+\.[0-9]+$ ]]; then
        MEAN_STD=$(awk -v a="${ACCS[0]}" -v b="${ACCS[1]}" -v c="${ACCS[2]}" 'BEGIN {
            mean = (a + b + c) / 3
            std  = sqrt(((a-mean)^2 + (b-mean)^2 + (c-mean)^2) / 2)
            printf "%.2f ± %.2f%%", mean, std
        }')
    fi

    printf "  %-6s  %-10s  %-10s  %-10s  %-14s\n" \
        "Q=$Q" \
        "${ACCS[0]:+${ACCS[0]}%}" \
        "${ACCS[1]:+${ACCS[1]}%}" \
        "${ACCS[2]:+${ACCS[2]}%}" \
        "$MEAN_STD"
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All $TOTAL runs completed successfully."
else
    echo "Failed runs (${#FAILED[@]}): ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
