#!/bin/bash
# Single rerun: HMDB51 Chebyshev full-fusion split 3.
# The only missing run from the ortho basis ablation.
#
# Usage:
#   cd vnn
#   bash chebyshev_hmdb51_split3.sh

set -euo pipefail

GPUS="4,5,6,7"
NPROC=4
MASTER_PORT=29505

RUN_NAME="hmdb51_lvn_chebyshev_fusion_nt4_ns2_split3"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_NAME}.log"

echo "Chebyshev HMDB51 split3 rerun — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  run_name: $RUN_NAME"
echo "  log:      $LOG"
echo ""

{
    echo "=== $RUN_NAME"
    echo "START: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
} > "$LOG" 2>&1

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPUS" \
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_par.py \
    --dataset     hmdb51 \
    --model       lvn_chebyshev_fusion \
    --n_lag       4 \
    --n_lag_s     2 \
    --split       3 \
    --epochs      100 \
    --batch_size  8 \
    --lr          4e-4 \
    --num_workers 4 \
    --wandb_group ortho_ablation_additive \
    --run_name    "$RUN_NAME" \
    --seed        42 \
    >> "$LOG" 2>&1

STATUS=$?
{
    echo ""
    echo "================================================================"
    echo "END: $(date '+%Y-%m-%d %H:%M:%S')  exit=$STATUS"
} >> "$LOG" 2>&1

if [ $STATUS -eq 0 ]; then
    ACC=$(grep -oP "(?<=Top-1: )\d+\.\d+" "$LOG" 2>/dev/null | tail -1 || true)
    [ -n "$ACC" ] && echo "Done — test acc: ${ACC}%" || echo "Done"
else
    echo "FAILED (exit $STATUS) — see $LOG"
    exit 1
fi
