#!/usr/bin/env bash
# ==============================================================================
# Re-run MiniMind small (hidden=768, layers=8) with fixed initialization order.
#
# Default run:
#   seeds:    42 123 2026
#   variants: s1-s13
#
# Outputs are intentionally separated from historical pretrain_v2 results:
#   weights/final/minimind-small-fixedinit-randsplit/seed42/s1_768.pth
#   weights/resume/minimind-small-fixedinit-randsplit/seed42/s1_768_resume.pth
#   logs/minimind-small-fixedinit/<run_id>/seed42/s1.log
#
# Examples:
#   bash scripts/experiments/run_minimind_small_fixedinit.sh
#   SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_minimind_small_fixedinit.sh
#   RUN_EVAL=0 MAX_STEPS=10 bash scripts/experiments/run_minimind_small_fixedinit.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
PY="$TORCH24_PREFIX/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[fixedinit-small] 找不到 torch24 Python: $PY"
    exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42 123 2026}"
VARIANTS="${VARIANTS:-s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13}"
WEIGHT_PREFIX_BASE="${WEIGHT_PREFIX_BASE:-pretrain_v2_fixedinit_randsplit}"
WEIGHT_NAMESPACE="${WEIGHT_NAMESPACE:-minimind-small-fixedinit-randsplit}"
FLAT_WEIGHT_NAMES="${FLAT_WEIGHT_NAMES:-0}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/$WEIGHT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/$WEIGHT_NAMESPACE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/minimind-small-fixedinit/$RUN_ID}"
RUN_LOG="$LOG_ROOT/run.log"
DATA_PATH="${DATA_PATH:-dataset/minimind/pretrain_t2t.jsonl}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-../$DATA_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-../model}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$DATA_PATH}"
EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-model}"
SPLIT_SEED="${SPLIT_SEED:-20260602}"
EVAL_SPLIT_RATIO="${EVAL_SPLIT_RATIO:-0.01}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-$ROOT/data_splits/minimind_pretrain_t2t_random_eval_${EVAL_SPLIT_RATIO}_seed${SPLIT_SEED}.json}"

RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-224}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-0}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MODE="${EVAL_MODE:-variants}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "[fixedinit-small] start at $(date '+%F %T')"
echo "  ROOT               = $ROOT"
echo "  GPUS               = $GPUS"
echo "  SEEDS              = $SEEDS"
echo "  VARIANTS           = $VARIANTS"
echo "  WEIGHT_PREFIX_BASE = $WEIGHT_PREFIX_BASE"
echo "  WEIGHT_NAMESPACE   = $WEIGHT_NAMESPACE"
echo "  FLAT_WEIGHT_NAMES  = $FLAT_WEIGHT_NAMES"
echo "  LOG_ROOT           = $LOG_ROOT"
echo "  FINAL_ROOT         = $FINAL_ROOT"
echo "  RESUME_ROOT        = $RESUME_ROOT"
echo "  DATA_PATH          = $TRAIN_DATA_PATH"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  SPLIT_MANIFEST     = $SPLIT_MANIFEST_PATH"
echo "  EVAL_SPLIT_RATIO   = $EVAL_SPLIT_RATIO"
echo "  SPLIT_SEED         = $SPLIT_SEED"
echo "  HIDDEN_SIZE        = 768"
echo "  NUM_HIDDEN_LAYERS  = 8"
echo "  RANK               = $RANK"
echo "  LM_HEAD_BIAS       = $LM_HEAD_BIAS"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  ACCUMULATION_STEPS = $ACCUMULATION_STEPS"
echo "  EPOCHS             = $EPOCHS"
echo "  LEARNING_RATE      = $LEARNING_RATE"
echo "  FROM_RESUME        = $FROM_RESUME"
echo "  MAX_STEPS          = $MAX_STEPS"
echo "  RUN_EVAL           = $RUN_EVAL"
echo "  EVAL_MODE          = $EVAL_MODE"
echo "================================================================"

"$PY" scripts/data/create_split_manifest.py \
    --data_path "$EVAL_DATA_PATH" \
    --output "$SPLIT_MANIFEST_PATH" \
    --eval_ratio "$EVAL_SPLIT_RATIO" \
    --seed "$SPLIT_SEED"

for seed in $SEEDS; do
    weight_prefix=""
    seed_save_dir="$FINAL_ROOT/seed${seed}"
    seed_resume_dir="$RESUME_ROOT/seed${seed}"
    if [[ "$FLAT_WEIGHT_NAMES" == "1" ]]; then
        weight_prefix="${WEIGHT_PREFIX_BASE}_seed${seed}"
        seed_save_dir="$ROOT/weights/final"
        seed_resume_dir="$ROOT/weights/resume"
    fi
    seed_log_dir="$LOG_ROOT/seed${seed}"
    mkdir -p "$seed_log_dir" "$seed_save_dir" "$seed_resume_dir"

    echo
    echo "----------------------------------------------------------------"
    echo "[fixedinit-small] seed=$seed weight_prefix=$weight_prefix"
    echo "  log_dir=$seed_log_dir"
    echo "  final dir=$seed_save_dir"
    echo "  resume dir=$seed_resume_dir"
    echo "----------------------------------------------------------------"

    GPUS="$GPUS" \
    VARIANTS="$VARIANTS" \
    SEED="$seed" \
    WEIGHT_PREFIX="$weight_prefix" \
    LOG_DIR="$seed_log_dir" \
    DATA_PATH="$TRAIN_DATA_PATH" \
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
    bash "$ROOT/scripts/train/train_minimind_small_variants.sh"

    if [[ "$RUN_EVAL" == "1" ]]; then
        echo
        echo "[fixedinit-small] eval seed=$seed at $(date '+%F %T')"
        PY="$PY" \
        GPU_LIST="$GPUS" \
        EVAL_MODE="$EVAL_MODE" \
        VARIANTS="$VARIANTS" \
        WEIGHT_PREFIX="$weight_prefix" \
        SAVE_DIR="$seed_save_dir" \
        DATA_PATH="$EVAL_DATA_PATH" \
        TOKENIZER_PATH="$EVAL_TOKENIZER_PATH" \
        OUTPUT_DIR="$seed_log_dir" \
        SUMMARY_DIR="$LOG_ROOT" \
        SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_PATH" \
        TAIL_RATIO=0 \
        HIDDEN_SIZE=768 \
        NUM_HIDDEN_LAYERS=8 \
        NUM_ATTENTION_HEADS=8 \
        NUM_KEY_VALUE_HEADS=4 \
        EMBEDDING_VARIANT_RANK="$RANK" \
        LM_HEAD_BIAS="$LM_HEAD_BIAS" \
        MAX_SEQ_LEN="$MAX_SEQ_LEN" \
        EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
        EVAL_NUM_WORKERS="$EVAL_NUM_WORKERS" \
        bash "$ROOT/scripts/eval/eval_parallel.sh"
    fi
done

echo
echo "================================================================"
echo "[fixedinit-small] done at $(date '+%F %T')"
echo "  log_root=$LOG_ROOT"
echo "================================================================"
