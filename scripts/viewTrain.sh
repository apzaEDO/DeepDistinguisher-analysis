#!/usr/bin/env bash
# train_goppa.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ------------------------------------------------------------------
# Train the code-distance model on a chosen Goppa dataset.
# ------------------------------------------------------------------

set -euo pipefail

# cleanup() {
#   echo "[cleanup] restoring clocks/fan control..."
#   sudo /usr/bin/jetson_clocks --restore || true
#   sudo systemctl restart nvfancontrol || true
# }

# trap cleanup EXIT INT TERM

# sudo /usr/bin/jetson_clocks --store
# sudo nvpmodel -m 1

sudo nvpmodel -m 1
sudo systemctl restart nvfancontrol

# ── tweak-once-use-everywhere parameters ──────────────────────────
GPU_ID=${GPU_ID:-0,1}            # override with:  GPU_ID=1 ./train_goppa.sh …
TASK="view-goppa"

CODE_LEN=1024
M_ALT=10
T_ALT=3
Q=2

TRAIN_SAMPLES=19500
EVAL_SAMPLES=500
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=16
VAL_EVERY=1000
LOG_EVERY=100
NUM_TRAIN_EPOCH=4
REPR="A"
CODE_LEN="${1:?Usage: $0 <CODE_LEN>}"
T_ALT="${2:?Usage: $0 <T_ALT>}"
REPR="${3:?Usage: $0 <H>}"
NUM_VIEWS="${4:?Usage :$0<NUM_VIEWS}"
DATA_ROOT="./data/dataset_goppa_${CODE_LEN}_H5"



# ── training call ─────────────────────────────────────────────────
#echo "🚀  Starting training on GPU ${GPU_ID}"
DUMP_PATH="${DUMP_PATH:-$PWD/checkpoint}"
mkdir -p "$DUMP_PATH"
#python train.py \

CUDA_VISIBLE_DEVICES=$GPU_ID \
#torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
python train.py \
    --task "$TASK" \
    --dump_path "$DUMP_PATH" \
    --exp_id "${REPR}_model_Goppa_N${CODE_LEN}_T${T_ALT}_M${M_ALT}" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --eval_samples "$EVAL_SAMPLES" \
    --train_samples "$TRAIN_SAMPLES" \
    --val_every "$VAL_EVERY" \
    --log_every "$LOG_EVERY" \
    --code_len "$CODE_LEN" \
    --data_path "./data/dataset_goppa_<codelen>_H5/A_goppa_nmt_<codelen>_<malt>_<talt>/dataset_10K.h5" \
    --random_data_path "./data/dataset_random_<codelen>_H5/A_random_nmt_<codelen>_<malt>_<talt>_/dataset_10K.h5" \
    --m_alt "$M_ALT" \
    --t_alt "$T_ALT" \
    --tqdm True \
    --Q "$Q" \
    --num_train_epochs "$NUM_TRAIN_EPOCH" \
    --representation "$REPR" \
    --compile False \
    --patch_cols 20 \
    --patch_rows 300 \
    --view_determinist  False\
    --eval_num_views $NUM_VIEWS

echo "✅  Training finished."