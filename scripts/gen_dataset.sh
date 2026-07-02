#!/usr/bin/env bash
# run_pipeline.sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ------------------------------------------------------------------
# Generate and collect datasets for the “goppa” and “random” codes.
# ------------------------------------------------------------------

set -euo pipefail

# ── shared parameters ─────────────────────────────────────────────
NUM_WORKERS=10
N_SAMPLES=10000
CODE_LEN=64
T_ALT=3
M_ALT=6
SAVE_EVERY=1000
DUMP_PATH="./data"
Q=2

# ── helper: run a code family end-to-end ───────────────────────────
run_dataset () {
  local CODE="$1"                         # "goppa" or "random"
  local EXP_NAME="dataset_${CODE}_${CODE_LEN}"
  local DATA_DIR="${DUMP_PATH}/${EXP_NAME}"

  echo "=== [${CODE}] Generating → ${EXP_NAME} ==="
  python -m scripts.generate_data \
         --code        "$CODE" \
         --num_workers "$NUM_WORKERS" \
         --n_samples   "$N_SAMPLES" \
         --code_len    "$CODE_LEN" \
         --t_alt       "$T_ALT" \
         --save_every  "$SAVE_EVERY" \
         --m_alt       "$M_ALT" \
         --dump_path   "$DUMP_PATH" \
         --exp_name    "$EXP_NAME" \
         --Q           "$Q"

  echo "=== [${CODE}] Collecting  → ${DATA_DIR} ==="
  python -m scripts.collect_data \
         --data_path   "$DATA_DIR" \
         --n_samples   "$N_SAMPLES" \
         --code        "$CODE" \
         --code_len    "$CODE_LEN" \
         --m_alt       "$M_ALT" \
         --t_alt       "$T_ALT" \
         --Q           "$Q"
}

# ── main loop ─────────────────────────────────────────────────────
for CODE in goppa random; do
  run_dataset "$CODE"
done

echo "🎉  All datasets generated and collected."