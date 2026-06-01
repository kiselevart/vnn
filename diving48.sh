#!/bin/bash
# diving48.sh — VNN ablation on Diving48 (single official train/test split)
#
# Data setup (do once before running):
#   1. Download Diving48 v2 from http://www.svcl.ucsd.edu/projects/resound/dataset.html
#      Extract so that vnn/data/diving48/ contains:
#        diving48_v2_train.json or Diving48_V2_train.json
#        diving48_v2_test.json or Diving48_V2_test.json
#        rgb/<vid_name>.mp4           (official RGB clips), or
#        rgb/<vid_name>/<frame>.jpg   (pre-extracted RGB frames)
#      The optical flow archive is NOT needed — flow is recomputed from RGB
#      using the same Farneback method as UCF101/HMDB51 for consistency.
#   2. Frame extraction/resizing + flow computation runs automatically on first launch.
#
# Runs: VNN multiplicative, VNN additive, SmallR3D, R3D-18 (no cubic, seed 42)
# Runtime: ~2-3h per run on 4 GPUs (~10h total)
# Launch: cd vnn && bash diving48.sh

set -e

GPUS="0,1,2,3"
PORT=29507
EPOCHS=100
SEED=42
GROUP="diving48_ablation"

run() {
    local model=$1 run_name=$2
    shift 2
    echo ""
    echo "==> $run_name  (model=$model)"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$GPUS \
    torchrun --nproc_per_node=4 --master_port=$PORT train_par.py \
        --dataset diving48 \
        --model "$model" \
        --epochs $EPOCHS \
        --batch_size 8 \
        --lr 4e-4 \
        --seed $SEED \
        --disable_cubic \
        --wandb_group "$GROUP" \
        --run_name "$run_name" \
        "$@" \
        2>&1 | tee logs/${run_name}.log
}

mkdir -p logs

# Core ablation: multiplicative vs additive (mirrors fusion_ablation_no_cubic protocol)
run vnn_fusion_ho          "diving48_vnn_fusion_ho_seed42_no_cubic"
run vnn_additive_fusion_ho "diving48_vnn_additive_fusion_ho_seed42_no_cubic"

# Parameter-matched baseline (~6.2M params, same as LVN/ortho)
run small_r3d "diving48_small_r3d_seed42"

# Standard baseline (full R3D-18, ~33M params)
run r3d "diving48_r3d18_seed42"

echo ""
echo "All Diving48 runs complete."
