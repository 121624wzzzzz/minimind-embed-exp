#!/usr/bin/env bash
# ==============================================================================
# FineEdu GPT-2-tokenizer experiment entrypoint.
#
# Defaults:
#   seeds:    42 123 2026
#   variants: s1,s2,s3,s6,s12
#
# Outputs:
#   weights/final/fineedu-gpt2/seed42/s1_768.pth
#   weights/resume/fineedu-gpt2/seed42/s1_768_resume.pth
#   logs/fineedu-gpt2/<run_id>/seed42/s1.log
#
# Examples:
#   bash scripts/experiments/run_fineedu_gpt2.sh
#   SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_fineedu_gpt2.sh
#   RUN_EVAL=0 MAX_STEPS=5 bash scripts/experiments/run_fineedu_gpt2.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
PY="$TORCH24_PREFIX/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[fineedu-exp] 找不到 torch24 Python: $PY"
    exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42 123 2026}"
VARIANTS="${VARIANTS:-s1,s2,s3,s6,s12}"
WEIGHT_NAMESPACE="${WEIGHT_NAMESPACE:-fineedu-gpt2}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/$WEIGHT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/$WEIGHT_NAMESPACE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fineedu-gpt2/$RUN_ID}"
RUN_LOG="$LOG_ROOT/run.log"

DATA_PATH="${DATA_PATH:-../dataset/fineweb_edu/gpt2_packed}"
TOKENIZER_PATH="${TOKENIZER_PATH:-gpt2}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-}"

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
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:1}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "[fineedu-exp] start at $(date '+%F %T')"
echo "  ROOT               = $ROOT"
echo "  GPUS               = $GPUS"
echo "  SEEDS              = $SEEDS"
echo "  VARIANTS           = $VARIANTS"
echo "  WEIGHT_NAMESPACE   = $WEIGHT_NAMESPACE"
echo "  LOG_ROOT           = $LOG_ROOT"
echo "  FINAL_ROOT         = $FINAL_ROOT"
echo "  RESUME_ROOT        = $RESUME_ROOT"
echo "  DATA_PATH          = $DATA_PATH"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  SPLIT_MANIFEST     = ${SPLIT_MANIFEST_PATH:-none}"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  EPOCHS             = $EPOCHS"
echo "  RUN_EVAL           = $RUN_EVAL"
echo "================================================================"

for seed in $SEEDS; do
    seed_save_dir="$FINAL_ROOT/seed${seed}"
    seed_resume_dir="$RESUME_ROOT/seed${seed}"
    seed_log_dir="$LOG_ROOT/seed${seed}"
    mkdir -p "$seed_log_dir" "$seed_save_dir" "$seed_resume_dir"

    echo
    echo "----------------------------------------------------------------"
    echo "[fineedu-exp] seed=$seed"
    echo "  log_dir=$seed_log_dir"
    echo "  final dir=$seed_save_dir"
    echo "  resume dir=$seed_resume_dir"
    echo "----------------------------------------------------------------"

    GPUS="$GPUS" \
    VARIANTS="$VARIANTS" \
    SEED="$seed" \
    WEIGHT_PREFIX="" \
    LOG_DIR="$seed_log_dir" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    SAVE_DIR="$seed_save_dir" \
    CHECKPOINT_DIR="$seed_resume_dir" \
    SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_PATH" \
    RANK="$RANK" \
    BATCH_SIZE="$BATCH_SIZE" \
    ACCUMULATION_STEPS="$ACCUMULATION_STEPS" \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    EPOCHS="$EPOCHS" \
    LEARNING_RATE="$LEARNING_RATE" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    NUM_WORKERS="$NUM_WORKERS" \
    FROM_RESUME="$FROM_RESUME" \
    MAX_STEPS="$MAX_STEPS" \
    LM_HEAD_BIAS="$LM_HEAD_BIAS" \
    TRAIN_SPLIT_RATIO="$TRAIN_SPLIT_RATIO" \
    GRAD_LOG_INTERVAL="$GRAD_LOG_INTERVAL" \
    GRAD_SAVE_TENSORS="$GRAD_SAVE_TENSORS" \
    RUN_EVAL="$RUN_EVAL" \
    EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
    EVAL_DEVICE="$EVAL_DEVICE" \
    bash "$ROOT/scripts/train/train_fineedu_gpt2_pretrain.sh"
done

echo
echo "================================================================"
echo "[fineedu-exp] done at $(date '+%F %T')"
echo "  log_root=$LOG_ROOT"
echo "================================================================"
