#!/usr/bin/env bash
set -euo pipefail

NUM_WORKERS=10
N_SAMPLES=10000
M_ALT=6
DUMP_PATH="./data"
Q=2

CODE_LEN="${1:?Usage: $0 <CODE_LEN> <T_ALT> <H>}"
T_ALT="${2:?Usage: $0 <CODE_LEN> <T_ALT> <H>}"
REPR="${3:?Usage: $0 <CODE_LEN> <T_ALT> <H>}"

SAVE_EVERY=$((N_SAMPLES / NUM_WORKERS))

run_dataset() {
  local CODE="$1"
  local EXP_NAME="dataset_${CODE}_${CODE_LEN}"
  local DATA_DIR="${DUMP_PATH}/${EXP_NAME}"

  echo "=== [${CODE}] Generating -> ${EXP_NAME} ==="
  python -m scripts.generate_data \
    --code "$CODE" \
    --num_workers "$NUM_WORKERS" \
    --n_samples "$N_SAMPLES" \
    --code_len "$CODE_LEN" \
    --t_alt "$T_ALT" \
    --save_every "$SAVE_EVERY" \
    --m_alt "$M_ALT" \
    --dump_path "$DUMP_PATH" \
    --exp_name "$EXP_NAME" \
    --standard_only True \
    --Q "$Q" \
    --representation "$REPR"

  echo "=== [${CODE}] Collecting -> ${DATA_DIR} ==="
  python -m scripts.collect_data \
    --data_path "$DATA_DIR" \
    --n_samples "$N_SAMPLES" \
    --code "$CODE" \
    --code_len "$CODE_LEN" \
    --m_alt "$M_ALT" \
    --t_alt "$T_ALT" \
    --Q "$Q" \
    --representation "$REPR"
}

#Choose between goppa or random
for CODE in goppa; do
  run_dataset "$CODE"
done

echo "All datasets generated and collected."




