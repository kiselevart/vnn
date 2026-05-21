#!/bin/bash
# Fusion ablation — compare multiplicative vs additive cross-stream fusion at Q=4.
#
# vnn_fusion_ho      : cat(rgb, flow, rgb*flow) → 288ch  — cross-stream product included
# vnn_additive_fusion_ho : cat(rgb, flow)        → 192ch  — product removed
#
# Purpose: isolate the effect of the cross-stream quadratic interaction term.
# Supports Proposition 2 (Jacobian instability from multiplicative fusion).
#
# Usage:
#   bash fusion_ablation.sh
#   bash fusion_ablation.sh 2>&1 | tee logs/fusion_ablation_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG — edit before running
# =============================================================================

GPUS="4,5,6,7"
NPROC=4

DATASET="ucf101"

# The two fusion variants to compare.
MODELS=("vnn_fusion_ho" "vnn_additive_fusion_ho")

# Fixed Q — use the sweet spot from the Q sweep.
Q=4

# Training hyperparams — match q_sweep.sh exactly for fair comparison.
EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4

# float32 required: vnn_fusion_ho has clamping but keep consistent with legacy runs.
EXTRA_ARGS="--no_amp"

SEED=42

WANDB_GROUP="fusion_ablation"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/fusion_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Fusion ablation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:  $DATASET"
echo "  Models:   ${MODELS[*]}"
echo "  Q:        $Q (fixed)"
echo "  GPUs:     $GPUS  (nproc=$NPROC)"
echo "  Epochs:   $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR"
echo "  Logs:     $LOG_DIR/"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/<run>.log"
echo "  grep -h 'Best\|exit=' $LOG_DIR/*.log"
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
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC train_par.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" train_par.py "$@" \
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
    RUN_NAME="${DATASET}_${MODEL}_Q${Q}_fusion_ablation"

    SEED_ARGS=()
    [ -n "$SEED" ] && SEED_ARGS=(--seed "$SEED")

    run_ddp "$RUN_NAME" \
        --dataset     "$DATASET" \
        --model       "$MODEL" \
        --Q           "$Q" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --wandb_group "$WANDB_GROUP" \
        --run_name    "$RUN_NAME" \
        "${SEED_ARGS[@]}" \
        $EXTRA_ARGS \
        || FAILED+=("$RUN_NAME")
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================== SUMMARY ==============================="
echo ""
echo "  Fusion ablation (Q=$Q, seed=$SEED):"
for MODEL in "${MODELS[@]}"; do
    RUN_NAME="${DATASET}_${MODEL}_Q${Q}_fusion_ablation"
    LOG="$LOG_DIR/${RUN_NAME}.log"
    if grep -q "exit=0" "$LOG" 2>/dev/null; then
        BEST=$(grep -oP "(?<=Best acc: )\S+" "$LOG" 2>/dev/null | tail -1 || echo "?")
        echo "    $MODEL  ✓  best val acc: $BEST"
    else
        echo "    $MODEL  ✗  (failed — check $LOG)"
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
echo "Next: compare test/acc + quad grad norms between multiplicative and additive fusion."
