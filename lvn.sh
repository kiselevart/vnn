#!/bin/bash
# LVN/fusion — all 3 splits of UCF101 and HMDB51.
#
# Model: lvn_laguerre_full_fusion (Laguerre basis on temporal + spatial dims)
#   --n_lag_t 4 : 4 temporal basis functions  (T=16 → 4 coefficients per kernel)
#   --n_lag_s 2 : 2 spatial basis functions   (3×3 kernels, center_spatial=True → rank ≤ 2)
# Fusion: additive (cat(rgb, flow) → 192ch)
# Basis LR: 3× multiplier on .coeff params (default laguerre_lr_mult=3.0)
#
# Usage:
#   cd vnn
#   bash lvn.sh
#   bash lvn.sh 2>&1 | tee logs/lvn_terminal.log

set -euo pipefail

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29504

MODEL="lvn_laguerre_full_fusion"
N_LAG_T=4
N_LAG_S=2
DATASETS=("ucf101" "hmdb51")
SPLITS=(1 2 3)

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
SEED=42
WANDB_GROUP="lvn_fusion_additive"

LOG_DIR="./logs/lvn_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "LVN/fusion (additive) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:    $MODEL  n_lag_t=$N_LAG_T  n_lag_s=$N_LAG_S"
echo "  Datasets: ${DATASETS[*]}  Splits: ${SPLITS[*]}"
echo "  GPUs:     $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR   Seed: $SEED"
echo "  Logs:     $LOG_DIR/"
echo ""

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
        local acc
        acc=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$log" 2>/dev/null | tail -1 || true)
        [ -n "$acc" ] && echo "done  (test acc: ${acc}%)" || echo "done"
    else
        echo "FAILED (exit $status) — see $log"
    fi

    return $status
}

FAILED=()

for DATASET in "${DATASETS[@]}"; do
    echo "--- Dataset: $DATASET ---"
    for SPLIT in "${SPLITS[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_nt${N_LAG_T}_ns${N_LAG_S}_split${SPLIT}"
        run_ddp "$RUN_NAME" \
            --dataset     "$DATASET" \
            --model       "$MODEL" \
            --n_lag_t     "$N_LAG_T" \
            --n_lag_s     "$N_LAG_S" \
            --split       "$SPLIT" \
            --epochs      "$EPOCHS" \
            --batch_size  "$BATCH_SIZE" \
            --lr          "$LR" \
            --num_workers "$NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP" \
            --run_name    "$RUN_NAME" \
            --seed        "$SEED" \
            || FAILED+=("$RUN_NAME")
    done
    echo ""
done

echo "=============================== SUMMARY ==============================="
for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "  $DATASET — LVN/fusion (n_lag_t=$N_LAG_T, n_lag_s=$N_LAG_S, additive):"
    printf "  %-8s  %-10s\n" "Split" "Test acc"
    printf "  %-8s  %-10s\n" "--------" "----------"

    ACCS=()
    for SPLIT in "${SPLITS[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_nt${N_LAG_T}_ns${N_LAG_S}_split${SPLIT}"
        LOG="$LOG_DIR/${RUN_NAME}.log"
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || echo "")
            [ -n "$ACC" ] && ACCS+=("$ACC")
            printf "  %-8s  %-10s\n" "split $SPLIT" "${ACC:+${ACC}%}"
        else
            printf "  %-8s  %-10s\n" "split $SPLIT" "FAILED"
        fi
    done

    if [ ${#ACCS[@]} -eq 3 ]; then
        MEAN=$(awk "BEGIN {printf \"%.2f\", (${ACCS[0]} + ${ACCS[1]} + ${ACCS[2]}) / 3}")
        STD=$(awk "BEGIN {
            a=${ACCS[0]}; b=${ACCS[1]}; c=${ACCS[2]}
            m=(a+b+c)/3
            printf \"%.2f\", sqrt(((a-m)^2+(b-m)^2+(c-m)^2)/2)
        }")
        printf "  %-8s  %s\n" "mean±std" "${MEAN} ± ${STD}%"
    fi
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
