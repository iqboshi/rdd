#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Network acceleration for GitHub/HF download paths (if available on host).
source /etc/network_turbo >/dev/null 2>&1 || true

export KMP_DUPLICATE_LIB_OK=TRUE
export HF_HUB_DISABLE_XET=1

PYTHON_EXE="${PYTHON_EXE:-/root/miniconda3/bin/python}"
SAVE_DIR="${SAVE_DIR:-outputs/ablation_aux_a8_lto_v1_q128}"
RESUME_CKPT="${RESUME_CKPT:-outputs/ablation_aux_a8_lto_v1_q128/exp_20260319_121144/checkpoints/latest.pth}"
BACKBONE_CKPT="${BACKBONE_CKPT:-pretrain_riceseg/outputs/exp_20260307_124458/checkpoints/best_backbone_for_instance.pth}"
EPOCHS="${EPOCHS:-95}"

# RTX 5090 + large-CPU host tuned defaults (override by env vars when needed).
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:---pretrained}"

exec "$PYTHON_EXE" -u train-v5.py \
  --save_dir "$SAVE_DIR" \
  --epochs "$EPOCHS" \
  --resume "$RESUME_CKPT" \
  --pretrained_backbone_path "$BACKBONE_CKPT" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --persistent_workers \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --vis_every 5 \
  --tqdm_postfix_interval 50 \
  --input_size 512 \
  --num_queries 128 \
  --enable_aux_heads \
  --amp \
  "$PRETRAINED_FLAG" \
  --device cuda \
  --w_cls 0.5 --w_mask 2.0 --w_dice 2.0 \
  --w_center 0.15 --w_offset 0.01 \
  --w_vote_consistency 0.005 \
  --w_separation 0.08 --w_repulsion 0.015 \
  --w_conflict 0.05 --w_affinity 0.03 --w_overlap_excl 0.02 \
  --w_order 0.05 --order_min_dy 4.0 --order_pair_max_dist 180.0 --order_max_pairs 4096 \
  --enable_patch_scale_weighting \
  --patch_scale_weight_512 1.0 --patch_scale_weight_768 1.8 --patch_scale_weight_1024 3.5
