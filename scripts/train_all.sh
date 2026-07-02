#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
TASK="code-dist-all-goppa"

M_ALT=6
Q=2
N_MAX=64

TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=2000
VAL_EVERY=1000
LOG_EVERY=100
TAILLE=4200000
REPR="A"


DUMP_PATH="${DUMP_PATH:-$PWD/checkpoint}"
mkdir -p "$DUMP_PATH"


for T_ALT in $(seq 2 2); do
    borne_inf=$((T_ALT * 8))

    if (( borne_inf > N_MAX )); then
        echo "Skip t=$T_ALT (borne_inf=$borne_inf > N_MAX=$N_MAX)"
        continue
    fi

    param_sets=""
    found_any=0

    for n in $(seq "$borne_inf" 8 "$N_MAX"); do
        subfolder="${REPR}_goppa_nmt_${n}_${M_ALT}_${T_ALT}"
        path="./data/dataset_goppa_${n}_H5/${subfolder}/dataset.h5"

        if [[ -f "$path" ]]; then
            if [[ -z "$param_sets" ]]; then
                param_sets="${n},${M_ALT},${T_ALT}"
            else
                param_sets="${param_sets};${n},${M_ALT},${T_ALT}"
            fi
            found_any=1
        else
            echo "Missing dataset for t=$T_ALT, n=$n -> $path"
        fi
    done

    if (( found_any == 0 )); then
        echo "No dataset found for t=$T_ALT, skipping."
        continue
    fi

    echo "--------------------------------------------------"
    echo "Training for t=$T_ALT"
    echo "param_sets=$param_sets"
    echo "--------------------------------------------------"
    NUM_TRAIN_EPOCH=$(($T_ALT*2))

    BORNE_INF=$((T_ALT * 8))
    MAX_N=$((2 ** M_ALT))
    NB_N=$((((MAX_N - BORNE_INF) / 8) + 1))
    SAMPLES_PER_N=$((4200000/NB_N))
    TOTAL_SAMPLES=$((4200000*2))
    NUM_EPOCHS=$((T_ALT*2))
    TRAIN_SAMPLES=$((TOTAL_SAMPLES * 99 / 100))
    EVAL_SAMPLES=$((TOTAL_SAMPLES - TRAIN_SAMPLES))

    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python train.py \
        --task "$TASK" \
        --dump_path "$DUMP_PATH" \
        --param_sets "$param_sets" \
        --exp_id "${REPR}_model_GoppaAll_Nmax${N_MAX}_T${T_ALT}_M${M_ALT}" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --eval_samples "$EVAL_SAMPLES" \
        --train_samples "$TRAIN_SAMPLES" \
        --val_every "$VAL_EVERY" \
        --log_every "$LOG_EVERY" \
        --code_len "$N_MAX" \
        --data_path "./data/dataset_goppa_<codelen>_H5/${REPR}_goppa_nmt_<codelen>_${M_ALT}_${T_ALT}/dataset.h5" \
        --m_alt "$M_ALT" \
        --t_alt "$T_ALT" \
        --tqdm True \
        --Q "$Q" \
        --num_train_epochs 1 \
        --representation "$REPR"

    echo "Done for t=$T_ALT"
done

echo "All trainings finished."