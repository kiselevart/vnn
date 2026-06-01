#!/bin/bash
# Q sweep — splits 2 and 3 of UCF101.
#
# Companion to q_sweep.sh, which already ran split 1 (seed=42, float32).
# Results from split 1:
#   Q=1  45.14%  Q=2  45.69%  Q=4  46.19%  Q=8  43.92%  Q=16  45.19%
#
# Running splits 2 and 3 gives a 3-split mean ± std for the Q ablation table,
# replacing the single-point split-1 numbers in the paper.
#
# Config is identical to q_sweep.sh: vnn_legacy_fusion, float32, seed=42, 4-GPU DDP.
#
# Usage:
#   cd vnn
#   bash q_sweep_splits.sh
#   bash q_sweep_splits.sh 2>&1 | tee logs/q_sweep_splits_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG — keep identical to q_sweep.sh except SPLITS
# =============================================================================

GPUS="4,5,6,7"
NPROC=4

DATASET="ucf101"
MODELS=("vnn_legacy_fusion")
Q_VALUES=(1 2 4 8 16)
SPLITS=(2 3)

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
EXTRA_ARGS="--no_amp"

SEED=42
MASTER_PORT=29500
WANDB_GROUP="q_sweep_splits"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/q_sweep_splits_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Q sweep (splits 2+3) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:  $DATASET"
echo "  Models:   ${MODELS[*]}"
echo "  Q values: ${Q_VALUES[*]}"
echo "  Splits:   ${SPLITS[*]}"
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
# Sweep
# =============================================================================

FAILED=()

for MODEL in "${MODELS[@]}"; do
    echo "--- Model: $MODEL ---"
    for SPLIT in "${SPLITS[@]}"; do
        echo "  -- Split $SPLIT --"
        for Q in "${Q_VALUES[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_Q${Q}_split${SPLIT}"
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
done

# =============================================================================
# Summary — per-Q results and 3-split averages (split 1 hardcoded from prior run)
# =============================================================================

# Split 1 test accs from logs/q_sweep_20260520_211547/
declare -A SPLIT1
SPLIT1[1]="45.14"
SPLIT1[2]="45.69"
SPLIT1[4]="46.19"
SPLIT1[8]="43.92"
SPLIT1[16]="45.19"

echo "=============================== SUMMARY ==============================="
echo ""
for MODEL in "${MODELS[@]}"; do
    echo "  $MODEL — test acc by Q and split:"
    echo ""
    printf "  %-6s  %-10s  %-10s  %-10s  %-10s\n" "Q" "split 1" "split 2" "split 3" "mean"
    printf "  %-6s  %-10s  %-10s  %-10s  %-10s\n" "------" "----------" "----------" "----------" "----------"

    for Q in "${Q_VALUES[@]}"; do
        S1="${SPLIT1[$Q]}"
        S2=""
        S3=""

        for SPLIT in "${SPLITS[@]}"; do
            RUN_NAME="${DATASET}_${MODEL}_Q${Q}_split${SPLIT}"
            LOG="$LOG_DIR/${RUN_NAME}.log"
            if grep -q "exit=0" "$LOG" 2>/dev/null; then
                ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || echo "")
                [ "$SPLIT" -eq 2 ] && S2="$ACC"
                [ "$SPLIT" -eq 3 ] && S3="$ACC"
            else
                [ "$SPLIT" -eq 2 ] && S2="FAILED"
                [ "$SPLIT" -eq 3 ] && S3="FAILED"
            fi
        done

        # Compute mean if all three splits have numeric values
        MEAN=""
        if [[ "$S1" =~ ^[0-9]+\.[0-9]+$ ]] && [[ "$S2" =~ ^[0-9]+\.[0-9]+$ ]] && [[ "$S3" =~ ^[0-9]+\.[0-9]+$ ]]; then
            MEAN=$(awk "BEGIN {printf \"%.2f\", ($S1 + $S2 + $S3) / 3}")
        fi

        printf "  %-6s  %-10s  %-10s  %-10s  %-10s\n" \
            "Q=$Q" "${S1:+${S1}%}" "${S2:+${S2}%}" "${S3:+${S3}%}" "${MEAN:+${MEAN}%}"
    done
    echo ""
done

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
