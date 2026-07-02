set -euo pipefail

#This script train the model for all code length you write in param_sets

# ── tweak-once-use-everywhere parameters ──────────────────────────
GPU_ID=${GPU_ID:-0}          
TASK="code-dist-all-goppa"

CODE_LEN=64
M_ALT=6
T_ALT=4
Q=2

TRAIN_SAMPLES=950000
EVAL_SAMPLES=50000
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=2000
VAL_EVERY=1000
LOG_EVERY=100

CODE_LEN="${1:?Usage: $0 <CODE_LEN>}"
T_ALT="${2:?Usage: $0 <T_ALT>}"
REPR="${3:?Usage:$0<REPR>}"

NUM_TRAIN_EPOCH=10
# ── training call ─────────────────────────────────────────────────
echo "Starting training on GPU ${GPU_ID}"
DUMP_PATH="${DUMP_PATH:-$PWD/checkpoint}"
mkdir -p "$DUMP_PATH"
CUDA_VISIBLE_DEVICES=$GPU_ID \
python train.py \
    --task "$TASK" \
    --dump_path "$DUMP_PATH" \
    --param_sets "32,${M_ALT},${T_ALT};40,${M_ALT},${T_ALT};48,${M_ALT},${T_ALT};56,${M_ALT},${T_ALT};64,${M_ALT},${T_ALT}"\
    --exp_id ${REPR}_model_NGoppa_N${CODE_LEN}_T${T_ALT}_M${M_ALT} \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --val_batch_size   "$VAL_BATCH_SIZE" \
    --eval_samples     "$EVAL_SAMPLES" \
    --train_samples    "$TRAIN_SAMPLES" \
    --val_every        "$VAL_EVERY" \
    --log_every        "$LOG_EVERY" \
    --code_len         "$CODE_LEN" \
    --data_path        "./data/dataset_goppa_<codelen>_H5/${REPR}_goppa_nmt_<codelen>_<malt>_<talt>/dataset_100K.h5" \
    --m_alt            "$M_ALT" \
    --t_alt            "$T_ALT" \
    --tqdm             True \
    --Q                "$Q" \
    --num_train_epochs "$NUM_TRAIN_EPOCH" \
    --representation             "$REPR"
    

echo "Training finished."

