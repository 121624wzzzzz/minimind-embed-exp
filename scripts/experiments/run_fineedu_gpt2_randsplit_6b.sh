#!/usr/bin/env bash
# FineEdu GPT-2 6B fixed-random split experiment entrypoint.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="${TORCH24_PREFIX:-/home/wz/anaconda3/envs/torch24}"
PY="${PY:-$TORCH24_PREFIX/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[fineedu-randsplit] 找不到 Python: $PY" >&2
    exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42 123 2026}"
VARIANTS="${VARIANTS:-s1,s2,s3,s4,s5,s6,s7,s11,s12,s13}"
DATA_PATH="${DATA_PATH:-$ROOT/dataset/fineweb_edu/packed/gpt2_6b_seq340}"
TOKENIZER_PATH="${TOKENIZER_PATH:-gpt2}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-$ROOT/data_splits/fineedu_gpt2_6b_random_eval_0.01_seed20260602.json}"

WEIGHT_NAMESPACE="${WEIGHT_NAMESPACE:-fineedu-gpt2-randsplit-6b}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/$WEIGHT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/$WEIGHT_NAMESPACE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fineedu-gpt2-randsplit-6b/$RUN_ID}"
RUN_LOG="$LOG_ROOT/run.log"

RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-80}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MODE="${EVAL_MODE:-variants}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"

if [[ ! -f "$SPLIT_MANIFEST_PATH" ]]; then
    echo "[fineedu-randsplit] missing manifest: $SPLIT_MANIFEST_PATH" >&2
    exit 1
fi

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "[fineedu-randsplit] start at $(date '+%F %T')"
echo "  GPUS             = $GPUS"
echo "  SEEDS            = $SEEDS"
echo "  VARIANTS         = $VARIANTS"
echo "  DATA_PATH        = $DATA_PATH"
echo "  SPLIT_MANIFEST   = $SPLIT_MANIFEST_PATH"
echo "  WEIGHT_NAMESPACE = $WEIGHT_NAMESPACE"
echo "  LOG_ROOT         = $LOG_ROOT"
echo "================================================================"

for seed in $SEEDS; do
    seed_save_dir="$FINAL_ROOT/seed${seed}"
    seed_resume_dir="$RESUME_ROOT/seed${seed}"
    seed_log_dir="$LOG_ROOT/seed${seed}"
    mkdir -p "$seed_save_dir" "$seed_resume_dir" "$seed_log_dir"

    TORCH24_PREFIX="$TORCH24_PREFIX" \
    PY="$PY" \
    GPUS="$GPUS" \
    SEED="$seed" \
    VARIANTS="$VARIANTS" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    SAVE_DIR="$seed_save_dir" \
    CHECKPOINT_DIR="$seed_resume_dir" \
    WEIGHT_PREFIX="" \
    TRAIN_SPLIT_RATIO="1.0" \
    SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_PATH" \
    BATCH_SIZE="$BATCH_SIZE" \
    ACCUMULATION_STEPS="$ACCUMULATION_STEPS" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    EPOCHS="$EPOCHS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LM_HEAD_BIAS="$LM_HEAD_BIAS" \
    RANK="$RANK" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    NUM_WORKERS="$NUM_WORKERS" \
    FROM_RESUME="$FROM_RESUME" \
    MAX_STEPS="$MAX_STEPS" \
    GRAD_LOG_INTERVAL="$GRAD_LOG_INTERVAL" \
    GRAD_SAVE_TENSORS="$GRAD_SAVE_TENSORS" \
    SKIP_COMPLETED="$SKIP_COMPLETED" \
    LOG_DIR="$seed_log_dir" \
    bash "$ROOT/scripts/train/train_fineedu_gpt2_pretrain.sh"

    if [[ "$RUN_EVAL" == "1" ]]; then
        PY="$PY" \
        GPU_LIST="$GPUS" \
        EVAL_MODE="$EVAL_MODE" \
        VARIANTS="$VARIANTS" \
        WEIGHT_PREFIX="" \
        SAVE_DIR="$seed_save_dir" \
        DATA_PATH="$DATA_PATH" \
        TOKENIZER_PATH="$TOKENIZER_PATH" \
        OUTPUT_DIR="$seed_log_dir" \
        SUMMARY_DIR="$LOG_ROOT" \
        SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_PATH" \
        TAIL_RATIO=0 \
        LM_HEAD_BIAS="$LM_HEAD_BIAS" \
        MAX_SEQ_LEN="$MAX_SEQ_LEN" \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        EVAL_NUM_WORKERS="$EVAL_NUM_WORKERS" \
        bash "$ROOT/scripts/eval/eval_parallel.sh"
    fi
done

echo "================================================================"
echo "[fineedu-randsplit] done at $(date '+%F %T')"
echo "  log_root=$LOG_ROOT"
echo "================================================================"
