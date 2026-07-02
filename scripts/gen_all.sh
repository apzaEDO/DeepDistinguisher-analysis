#!/usr/bin/env bash
set -euo pipefail

#"This script allow you to generate all data necessary to train DeepDistinguisher on matrix representation A for t values 2-8"

NUM_WORKERS=10
DUMP_PATH="./data"
Q=2
REPR="A"

M_ALT=6

run_dataset() {
  local CODE="$1"
  local CODE_LEN="$2"
  local T_ALT="$3"
  local N_SAMPLES="$4"

  local SAVE_EVERY=$((N_SAMPLES / NUM_WORKERS))
  if (( SAVE_EVERY < 1 )); then
    SAVE_EVERY=1
  fi

  local EXP_NAME="dataset_${CODE}_${CODE_LEN}"
  local DATA_DIR="${DUMP_PATH}/${EXP_NAME}"

  echo "=== [${CODE}] Generating -> ${EXP_NAME} | n=${CODE_LEN}, t=${T_ALT}, samples=${N_SAMPLES} ==="

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

for t in $(seq 2 8); do
  borne_inf=$((t * 8))
  NB_N=$((((64 - borne_inf) / 8) + 1))

  if [ "$t" -lt 6 ]; then
    samples=$((4200000 / NB_N))
  else
    samples=$((20000000 / NB_N))
  fi

  echo "t=$t borne_inf=$borne_inf NB_N=$NB_N samples=$samples"

  for n in $(seq "$borne_inf" 8 64); do
    python ./scripts/make_dataset_all_goppa_parallel.py "$n" "$M_ALT" "$t" "$samples"
    run_dataset "random" "$n" "$t" "$samples"
  done
done

echo "All datasets generated and collected."