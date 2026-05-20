#!/bin/bash
# Q sweep — train vnn_legacy_fusion at multiple Q values sequentially on 4-GPU DDP.
# Purpose: validate Proposition 2 (subspace restriction) from the ablation paper.
#
# Each run uses all 4 GPUs via torchrun DDP. Jobs run one at a time (sequential),
# so this script ties up all 4 GPUs for the duration of the sweep.
#
# Usage:
#   bash q_sweep.sh
#   bash q_sweep.sh 2>&1 | tee logs/q_sweep_terminal.log   # also capture stdout

set -euo pipefail

# =============================================================================
# CONFIG — edit before running
# =============================================================================

# Which GPUs to use (must match --nproc_per_node below).
# GPU 5+6 have a broken P2P cross-socket link on this server — keep them
# together or separated, but always set NCCL_P2P_DISABLE=1.
GPUS="4,5,6,7"
NPROC=4

DATASET="ucf101"

# Models to sweep. vnn_legacy_fusion = no gates/shortcuts/cubic/clamping.
# Add "vnn_fusion_ho" to also benchmark the current higher-order model.
MODELS=("vnn_legacy_fusion")

# Q values to sweep over backbone quadratic rank.
Q_VALUES=(1 2 4 8 16)

# Training hyperparams.
# LR scales linearly with GPU count (1e-4 per GPU × NPROC).
EPOCHS=100
BATCH_SIZE=8           # per GPU; float32 uses 2× memory vs AMP, halve to compensate
LR="4e-4"             # 4 GPUs × 1e-4
NUM_WORKERS=4

# Legacy model has no output clamping — incompatible with float16 AMP (products overflow).
# Run in float32 to match the original paper's training setup.
EXTRA_ARGS="--no_amp"

# Random seed — set to an integer for reproducibility, or "" to disable.
SEED=42

# W&B group tag for this sweep (set --no_wandb below to disable W&B entirely).
WANDB_GROUP="q_sweep"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/q_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Q sweep — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Dataset:  $DATASET"
echo "  Models:   ${MODELS[*]}"
echo "  Q values: ${Q_VALUES[*]}"
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
        # Extract best val acc from log if present
        local best
        best=$(grep -oP "(?<=Best acc: )\S+" "$log" 2>/dev/null | tail -1 || true)
        [ -n "$best" ] && echo "done  (best val acc: $best)" || echo "done"
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

    for Q in "${Q_VALUES[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_Q${Q}"

        SEED_ARGS=()
        [ -n "$SEED" ] && SEED_ARGS=(--seed "$SEED")

        run_ddp "$RUN_NAME" \
            --dataset    "$DATASET" \
            --model      "$MODEL" \
            --Q          "$Q" \
            --epochs     "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr         "$LR" \
            --num_workers "$NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP" \
            --run_name   "$RUN_NAME" \
            "${SEED_ARGS[@]}" \
            $EXTRA_ARGS \
            || FAILED+=("$RUN_NAME")
    done

    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "  $MODEL:"
    for Q in "${Q_VALUES[@]}"; do
        RUN_NAME="${DATASET}_${MODEL}_Q${Q}"
        LOG="$LOG_DIR/${RUN_NAME}.log"
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            BEST=$(grep -oP "(?<=Best acc: )\S+" "$LOG" 2>/dev/null | tail -1 || echo "?")
            echo "    Q=$Q  ✓  best val acc: $BEST"
        else
            echo "    Q=$Q  ✗  (failed — check $LOG)"
        fi
    done
done

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All runs completed successfully."
else
    echo "Failed runs: ${FAILED[*]}"
    echo "Re-run failed jobs individually:"
    for r in "${FAILED[@]}"; do
        echo "  bash q_sweep.sh  # or extract the specific torchrun command from $LOG_DIR/${r}.log"
    done
    exit 1
fi

echo ""
echo "Logs: $LOG_DIR/"
echo "Next: compile Q vs val-acc table for the paper."
