#!/bin/bash
# Fusion ablation WITHOUT cubic — multiplicative vs additive cross-stream fusion.
#
# Identical to fusion_ablation.sh except --disable_cubic is set, so blocks 3/4
# and the fusion head run quadratic-only (Q auto-boosted to match parameter count).
#
# Why this is needed:
#   fusion_ablation.sh runs with cubic ON (default). The Q sweep was done on
#   vnn_legacy_fusion which has no cubic. Running fusion ablation with cubic ON
#   conflates the fusion mechanism (Prop 2) with the cubic interaction, making
#   cross-experiment comparison architecturally inconsistent. This script gives
#   the clean comparison: both models are quadratic-only, differing only in
#   whether rgb*flow cross-stream product is present.
#
# Run AFTER the current fusion_ablation.sh completes.
#
# Usage:
#   bash fusion_ablation_no_cubic.sh
#   bash fusion_ablation_no_cubic.sh 2>&1 | tee logs/fusion_ablation_no_cubic_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG — must match fusion_ablation.sh exactly except for cubic
# =============================================================================

GPUS="4,5,6,7"
NPROC=4

DATASET="ucf101"
MODELS=("vnn_fusion_ho" "vnn_additive_fusion_ho")

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4

# float32 + no cubic
EXTRA_ARGS="--no_amp --disable_cubic"

SEEDS=(42)
MASTER_PORT=29500
WANDB_GROUP="fusion_ablation_no_cubic"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/fusion_ablation_no_cubic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Fusion ablation (no cubic) — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:  $DATASET"
echo "  Models:   ${MODELS[*]}"
echo "  Cubic:    DISABLED (Q auto-boosted)"
echo "  Seeds:    ${SEEDS[*]}"
echo "  GPUs:     $GPUS  (nproc=$NPROC)  port=$MASTER_PORT"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
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
        local best
        best=$(grep -oP "(?<=Best acc: )\S+" "$log" 2>/dev/null | tail -1 || true)
        [ -n "$best" ] && echo "done  (best val acc: $best)" || echo "done"
    else
        echo "FAILED (exit $status) — see $log"
    fi

    return $status
}

# =============================================================================
# Runs
# =============================================================================

FAILED=()

for MODEL in "${MODELS[@]}"; do
    echo "--- Model: $MODEL ---"
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_no_cubic"

        run_ddp "$RUN_NAME" \
            --dataset     "$DATASET" \
            --model       "$MODEL" \
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
echo "  Fusion ablation — no cubic (seeds=${SEEDS[*]}):"
for MODEL in "${MODELS[@]}"; do
    echo "  $MODEL:"
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_seed${SEED}_no_cubic"
        LOG="$LOG_DIR/${RUN_NAME}.log"
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            BEST=$(grep -oP "(?<=Best acc: )\S+" "$LOG" 2>/dev/null | tail -1 || echo "?")
            echo "    seed=$SEED  ✓  best val acc: $BEST"
        else
            echo "    seed=$SEED  ✗  (failed — check $LOG)"
        fi
    done
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
echo "Next: compare with fusion_ablation.sh results — does cross-stream product"
echo "      help with or without cubic? Report both in Section 6.2."
