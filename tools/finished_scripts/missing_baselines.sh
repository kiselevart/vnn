#!/bin/bash
# missing_baselines.sh — fill in the $^\S$ and "---" gaps in Table 2.
#
# What this covers:
#   1. Large baselines (R3D, R(2+1)D, ResNet-50) — UCF-101 splits 2+3
#      (split 1 already done in finished_scripts/baselines.sh)
#   2. Large baselines (R3D, R(2+1)D, ResNet-50) — HMDB-51 all 3 splits
#   3. Two-stream I3D — HMDB-51 all 3 splits
#      (UCF-101 already done in finished_scripts/i3d_ucf_avg.sh)
#
# After this script, every cell in Table 2 has a 3-split mean ± std.
#
# Runtime estimate (4 GPUs):
#   Large baselines: ~2h/run × 12 runs = ~24h
#   I3D HMDB-51:     ~2.5h/run × 3 runs = ~7.5h
#   Total: ~31.5h sequential
#
# Usage:
#   cd vnn
#   bash missing_baselines.sh
#   bash missing_baselines.sh 2>&1 | tee logs/missing_baselines_terminal.log

set -euo pipefail

# =============================================================================
# CONFIG
# =============================================================================

GPUS="0,1,2,3"
NPROC=4
PORT_BASE=29509     # 29510 for I3D section
PORT_I3D=29510

EPOCHS=100
BATCH_SIZE=8
LR="4e-4"
NUM_WORKERS=4
SEED=42

WANDB_GROUP_BASELINES="baselines"
WANDB_GROUP_I3D="i3d_twostream_hmdb51"

# =============================================================================
# Setup
# =============================================================================

LOG_DIR="./logs/missing_baselines_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Missing baselines — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  GPUs: $GPUS  (nproc=$NPROC)"
echo "  Epochs: $EPOCHS   BS/GPU: $BATCH_SIZE   LR: $LR   Seed: $SEED"
echo "  Logs: $LOG_DIR/"
echo ""

FAILED=()

# =============================================================================
# Helper: train_par.py (RGB-only models)
# =============================================================================

run_ddp() {
    local run_name="$1"
    shift
    local log="$LOG_DIR/${run_name}.log"

    {
        echo "=== $run_name"
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$PORT_BASE train_par.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" --master_port="$PORT_BASE" train_par.py "$@" \
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
        FAILED+=("$run_name")
    fi
}

# =============================================================================
# Helper: train_i3d_two_stream.py
# =============================================================================

run_i3d() {
    local run_name="$1"
    shift
    local log="$LOG_DIR/${run_name}.log"

    {
        echo "=== $run_name"
        echo "CMD: NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$NPROC --master_port=$PORT_I3D tools/train_i3d_two_stream.py $*"
        echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "================================================================"
    } > "$log" 2>&1

    echo -n "  Running $run_name ... "

    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
        torchrun --nproc_per_node="$NPROC" --master_port="$PORT_I3D" \
        tools/train_i3d_two_stream.py "$@" \
        >> "$log" 2>&1
    local status=$?

    {
        echo ""
        echo "================================================================"
        echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$status"
    } >> "$log" 2>&1

    if [ $status -eq 0 ]; then
        local acc
        acc=$(grep "Top-1:" "$log" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+" | tail -1 || true)
        [ -n "$acc" ] && echo "done  (test acc: ${acc}%)" || echo "done"
    else
        echo "FAILED (exit $status) — see $log"
        FAILED+=("$run_name")
    fi
}

# =============================================================================
# 1. Large baselines — UCF-101 splits 2 and 3
#    (split 1 done: ucf101_{model}_seed42_baseline in finished_scripts/baselines.sh)
# =============================================================================

echo "=== Large baselines: UCF-101 splits 2+3 ==="
for SPLIT in 2 3; do
    for MODEL in r3d r2plus1d resnet50_frame_avg; do
        RUN="ucf101_${MODEL}_seed${SEED}_split${SPLIT}"
        run_ddp "$RUN" \
            --dataset     ucf101 \
            --model       "$MODEL" \
            --split       "$SPLIT" \
            --epochs      "$EPOCHS" \
            --batch_size  "$BATCH_SIZE" \
            --lr          "$LR" \
            --num_workers "$NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP_BASELINES" \
            --run_name    "$RUN" \
            --seed        "$SEED"
    done
done
echo ""

# =============================================================================
# 2. Large baselines — HMDB-51 all 3 splits
# =============================================================================

echo "=== Large baselines: HMDB-51 all 3 splits ==="
for SPLIT in 1 2 3; do
    for MODEL in r3d r2plus1d resnet50_frame_avg; do
        RUN="hmdb51_${MODEL}_seed${SEED}_split${SPLIT}"
        run_ddp "$RUN" \
            --dataset     hmdb51 \
            --model       "$MODEL" \
            --split       "$SPLIT" \
            --epochs      "$EPOCHS" \
            --batch_size  "$BATCH_SIZE" \
            --lr          "$LR" \
            --num_workers "$NUM_WORKERS" \
            --wandb_group "$WANDB_GROUP_BASELINES" \
            --run_name    "$RUN" \
            --seed        "$SEED"
    done
done
echo ""

# =============================================================================
# 3. Two-stream I3D — HMDB-51 all 3 splits
#    (UCF-101 done in finished_scripts/i3d_ucf_avg.sh)
# =============================================================================

echo "=== Two-stream I3D: HMDB-51 all 3 splits ==="
for SPLIT in 1 2 3; do
    RUN="hmdb51_i3d_twostream_seed${SEED}_split${SPLIT}"
    run_i3d "$RUN" \
        --dataset     hmdb51 \
        --split       "$SPLIT" \
        --run_name    "$RUN" \
        --epochs      "$EPOCHS" \
        --batch_size  "$BATCH_SIZE" \
        --lr          "$LR" \
        --num_workers "$NUM_WORKERS" \
        --wandb_group "$WANDB_GROUP_I3D"
done
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================== SUMMARY ==============================="
echo ""

mean_std() {
    awk -v a="$1" -v b="$2" -v c="$3" 'BEGIN {
        m = (a + b + c) / 3
        s = sqrt(((a-m)^2 + (b-m)^2 + (c-m)^2) / 2)
        printf "%.1f ± %.1f%%", m, s
    }'
}

acc_from_log() {
    grep -oP "(?<=Top-1: )\d+\.\d+" "$1" 2>/dev/null | tail -1 || echo ""
}

# Large baselines — UCF-101: combine split 1 (from baselines.sh) with splits 2+3
echo "Large baselines — UCF-101 (split 1 from previous run, 2+3 from this run):"
for MODEL in r3d r2plus1d resnet50_frame_avg; do
    # split 1 log was in the original baselines run; we can only see what's in $LOG_DIR
    ACCS=()
    for SPLIT in 2 3; do
        LOG="$LOG_DIR/ucf101_${MODEL}_seed${SEED}_split${SPLIT}.log"
        ACC=$(acc_from_log "$LOG")
        [ -n "$ACC" ] && ACCS+=("$ACC")
    done
    printf "  %-22s  splits 2+3: %s  %s\n" "$MODEL" "${ACCS[0]:-?}%" "${ACCS[1]:-?}%"
done
echo ""

# Large baselines — HMDB-51
echo "Large baselines — HMDB-51 (3-split mean ± std):"
for MODEL in r3d r2plus1d resnet50_frame_avg; do
    ACCS=()
    for SPLIT in 1 2 3; do
        LOG="$LOG_DIR/hmdb51_${MODEL}_seed${SEED}_split${SPLIT}.log"
        ACC=$(acc_from_log "$LOG")
        [ -n "$ACC" ] && ACCS+=("$ACC")
    done
    if [ ${#ACCS[@]} -eq 3 ]; then
        STAT=$(mean_std "${ACCS[0]}" "${ACCS[1]}" "${ACCS[2]}")
        printf "  %-22s  %s\n" "$MODEL" "$STAT"
    else
        printf "  %-22s  %d/3 splits completed\n" "$MODEL" "${#ACCS[@]}"
    fi
done
echo ""

# I3D HMDB-51
echo "Two-stream I3D — HMDB-51 (3-split mean ± std):"
ACCS=()
for SPLIT in 1 2 3; do
    LOG="$LOG_DIR/hmdb51_i3d_twostream_seed${SEED}_split${SPLIT}.log"
    ACC=$(acc_from_log "$LOG")
    [ -n "$ACC" ] && ACCS+=("$ACC")
done
if [ ${#ACCS[@]} -eq 3 ]; then
    STAT=$(mean_std "${ACCS[0]}" "${ACCS[1]}" "${ACCS[2]}")
    printf "  I3D two-stream  %s\n" "$STAT"
else
    printf "  I3D two-stream  %d/3 splits completed\n" "${#ACCS[@]}"
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
echo "Next: compute UCF-101 3-split means for large baselines by combining"
echo "  split 1 from baselines.sh logs with splits 2+3 above, then update Table 2."
