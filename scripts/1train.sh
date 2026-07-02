set -euo pipefail

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

GPU_ID=${GPU_ID:-0,1}           
TASK="code-dist-goppa"

CODE_LEN=64
M_ALT=6
T_ALT=2
Q=2

TRAIN_SAMPLES=19000
EVAL_SAMPLES=1000
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=8
VAL_EVERY=1000
LOG_EVERY=10
NUM_TRAIN_EPOCH=3
CODE_LEN="${1:?Usage: $0 <CODE_LEN>}"
T_ALT="${2:?Usage: $0 <T_ALT>}"
REPR="${3:?Usage: $0 <H>}"
DATA_ROOT="./data/dataset_goppa_${CODE_LEN}_H5"


DATA_PATH="${DATA_PATH:-$(ls -td ${DATA_ROOT}/${REPR}_goppa_nmt_${CODE_LEN}_${M_ALT}_${T_ALT}/dataset_10K.h5)}" 


DUMP_PATH="${DUMP_PATH:-$PWD/checkpoint}"
mkdir -p "$DUMP_PATH"

CUDA_VISIBLE_DEVICES=$GPU_ID \

python -m train.py \
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
    --data_path "$DATA_PATH" \
    --m_alt "$M_ALT" \
    --t_alt "$T_ALT" \
    --tqdm True \
    --Q "$Q" \
    --num_train_epochs "$NUM_TRAIN_EPOCH" \
    --representation "$REPR" \
    --compile False 

echo "Training finished."