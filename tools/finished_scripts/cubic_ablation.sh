#!/bin/bash
# Cubic interaction ablation — isolates the effect of the cubic term and its type.
#
# Compares three configurations of vnn_fusion_ho (multiplicative fusion, Q=4):
#
#   no_cubic   — cubic disabled; Q auto-boosted to match parameter count
#   symmetric  — cubic enabled, a²·b factorization  (2·Qc channels, default)
#   general    — cubic enabled, a·b·c factorization  (3·Qc channels)
#
# All other hyperparams identical. Results answer:
#   1. Does the cubic term help at all?         (no_cubic vs symmetric)
#   2. Does the factorization type matter?       (symmetric vs general)
#
# Run after fusion ablation and fusion_ablation_no_cubic complete, so you have
# the full 2×3 table: {multiplicative, additive} × {no_cubic, sym, gen}.
#
# Usage:
#   bash cubic_ablation.sh
#   bash cubic_ablation.sh 2>&1 | tee logs/cubic_ablation_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="4,5,6,7"
NPROC=4

DATASET="ucf101"
MODEL="vnn_fusion_ho"   # fixed: multiplicative fusion, isolates cubic effect

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4

SEEDS=(42)
MASTER_PORT=29500
WANDB_GROUP="cubic_ablation"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/cubic_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Cubic ablation — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:    $MODEL"
echo "  Dataset:  $DATASET"
echo "  Variants: no_cubic | symmetric | general"
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

for SEED in "${SEEDS[@]}"; do

    # 1. No cubic (Q auto-boosted for parameter parity)
    echo "--- no_cubic ---"
    RUN_NO_CUBIC="${DATASET}_${MODEL}_seed${SEED}_cubic_none"
    run_ddp "$RUN_NO_CUBIC" \
        --dataset      "$DATASET" \
        --model        "$MODEL" \
        --epochs       "$EPOCHS" \
        --batch_size   "$BATCH_SIZE" \
        --lr           "$LR" \
        --num_workers  "$NUM_WORKERS" \
        --wandb_group  "$WANDB_GROUP" \
        --run_name     "$RUN_NO_CUBIC" \
        --seed         "$SEED" \
        --no_amp \
        --disable_cubic \
        || FAILED+=("$RUN_NO_CUBIC")
    echo ""

    # 2. Cubic symmetric (a²·b) — default
    echo "--- cubic: symmetric ---"
    RUN_SYM="${DATASET}_${MODEL}_seed${SEED}_cubic_sym"
    run_ddp "$RUN_SYM" \
        --dataset      "$DATASET" \
        --model        "$MODEL" \
        --epochs       "$EPOCHS" \
        --batch_size   "$BATCH_SIZE" \
        --lr           "$LR" \
        --num_workers  "$NUM_WORKERS" \
        --wandb_group  "$WANDB_GROUP" \
        --run_name     "$RUN_SYM" \
        --seed         "$SEED" \
        --no_amp \
        --cubic_mode   symmetric \
        || FAILED+=("$RUN_SYM")
    echo ""

    # 3. Cubic general (a·b·c)
    echo "--- cubic: general ---"
    RUN_GEN="${DATASET}_${MODEL}_seed${SEED}_cubic_gen"
    run_ddp "$RUN_GEN" \
        --dataset      "$DATASET" \
        --model        "$MODEL" \
        --epochs       "$EPOCHS" \
        --batch_size   "$BATCH_SIZE" \
        --lr           "$LR" \
        --num_workers  "$NUM_WORKERS" \
        --wandb_group  "$WANDB_GROUP" \
        --run_name     "$RUN_GEN" \
        --seed         "$SEED" \
        --no_amp \
        --cubic_mode   general \
        || FAILED+=("$RUN_GEN")
    echo ""

done

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""
echo "  $MODEL — cubic ablation (seeds=${SEEDS[*]}):"
echo ""

for SEED in "${SEEDS[@]}"; do
    RUN_NO_CUBIC="${DATASET}_${MODEL}_seed${SEED}_cubic_none"
    RUN_SYM="${DATASET}_${MODEL}_seed${SEED}_cubic_sym"
    RUN_GEN="${DATASET}_${MODEL}_seed${SEED}_cubic_gen"

    for PAIR in "no_cubic:$RUN_NO_CUBIC" "symmetric:$RUN_SYM" "general:$RUN_GEN"; do
        LABEL="${PAIR%%:*}"
        RNAME="${PAIR##*:}"
        LOG="$LOG_DIR/${RNAME}.log"
        if grep -q "exit=0" "$LOG" 2>/dev/null; then
            BEST=$(grep -oP "(?<=Best acc: )\S+" "$LOG" 2>/dev/null | tail -1 || echo "?")
            echo "    seed=$SEED  $LABEL  ✓  best val acc: $BEST"
        else
            echo "    seed=$SEED  $LABEL  ✗  (failed — check $LOG)"
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
echo "Next: build 2×3 ablation table in paper:"
echo "  rows: {multiplicative fusion, additive fusion}"
echo "  cols: {no cubic, symmetric cubic, general cubic}"
