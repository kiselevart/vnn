#!/bin/bash
# Small VNN ablation — vnn_small_additive_fusion only.
#
# vnn_small_legacy_fusion is already complete (UCF101 45.3±2.1%, HMDB51 26.1±1.7%).
#
# Model:
#   vnn_small_additive_fusion — modern arch (gates, shortcuts), ch_per_kernel=4, Q=1, no cubic
#                               6.93 GFLOPs, 6.76M params
#
# 3 splits × UCF101 + HMDB51 = 6 runs total. ~2–2.5h/run on 4 GPUs → ~12–15h total.
#
# Usage:
#   cd vnn
#   bash small_vnn.sh
#   bash small_vnn.sh 2>&1 | tee logs/small_vnn_terminal.log

set -euo pipefail

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29508

MODELS=("vnn_small_additive_fusion")
DATASETS=("ucf101" "hmdb51")
SPLITS=(1 2 3)

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
SEED=42
WANDB_GROUP="small_vnn_ablation"

LOG_DIR="./logs/small_vnn_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Small VNN ablation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Models:   ${MODELS[*]}"
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

for MODEL in "${MODELS[@]}"; do
    echo "=== Model: $MODEL ==="
    for DATASET in "${DATASETS[@]}"; do
        echo "  --- Dataset: $DATASET ---"
        for SPLIT in "${SPLITS[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_split${SPLIT}"
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
                || FAILED+=("$RUN_NAME")
        done
        echo ""
    done
done

echo "=============================== SUMMARY ==============================="
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "  $MODEL:"
    printf "  %-10s  %-8s  %-10s\n" "Dataset" "Split" "Test acc"
    printf "  %-10s  %-8s  %-10s\n" "----------" "--------" "----------"

    for DATASET in "${DATASETS[@]}"; do
        ACCS=()
        for SPLIT in "${SPLITS[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_split${SPLIT}"
            LOG="$LOG_DIR/${RUN_NAME}.log"
            if grep -q "exit=0" "$LOG" 2>/dev/null; then
                ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || echo "")
                [ -n "$ACC" ] && ACCS+=("$ACC")
                printf "  %-10s  %-8s  %-10s\n" "$DATASET" "split $SPLIT" "${ACC:+${ACC}%}"
            else
                printf "  %-10s  %-8s  %-10s\n" "$DATASET" "split $SPLIT" "FAILED"
            fi
        done
        if [ ${#ACCS[@]} -eq 3 ]; then
            MEAN=$(awk "BEGIN {printf \"%.2f\", (${ACCS[0]} + ${ACCS[1]} + ${ACCS[2]}) / 3}")
            STD=$(awk "BEGIN {
                a=${ACCS[0]}; b=${ACCS[1]}; c=${ACCS[2]}
                m=(a+b+c)/3
                printf \"%.2f\", sqrt(((a-m)^2+(b-m)^2+(c-m)^2)/2)
            }")
            printf "  %-10s  %-8s  %s\n" "$DATASET" "mean±std" "${MEAN} ± ${STD}%"
        fi
        echo ""
    done
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi
echo ""
echo "Logs: $LOG_DIR/"
