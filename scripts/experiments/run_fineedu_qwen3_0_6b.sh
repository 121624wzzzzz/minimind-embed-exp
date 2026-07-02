#!/usr/bin/env bash
# ==============================================================================
# FineWeb-Edu 20B + Qwen3-0.6B config experiment entrypoint.
#
# Data:    FineWeb-Edu 20B tokens, packed seq_len=340, Qwen3 tokenizer
# Arch:    Qwen3-0.6B aligned (hidden=1024, layers=28, GQA 16/8, head_dim=128)
# Split:   fixed random manifest (1% eval, seed=20260602)
#
# Defaults:
#   seeds:    42 123 2026
#   variants: s1,s2,s3,s6,s12
#
# Outputs:
#   weights/final/fineedu-qwen3-0.6b/seed42/s1_1024.pth
#   weights/resume/fineedu-qwen3-0.6b/seed42/s1_1024_resume.pth
#   logs/fineedu-qwen3-0.6b/<run_id>/seed42/s1.log
#
# Examples:
#   bash scripts/experiments/run_fineedu_qwen3_0_6b.sh
#   SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_fineedu_qwen3_0_6b.sh
#   RUN_EVAL=0 MAX_STEPS=10 BATCH_SIZE=16 bash scripts/experiments/run_fineedu_qwen3_0_6b.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
PY="$TORCH24_PREFIX/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[fineedu-qwen3] 找不到 torch24 Python: $PY"
    exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42 123 2026}"
VARIANTS="${VARIANTS:-s1,s2,s3,s6,s12}"

# ---- 数据 & 切分 ----
DATA_PATH="${DATA_PATH:-../dataset/fineweb_edu/packed/qwen3_20b_seq340}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-dataset/fineweb_edu/packed/qwen3_20b_seq340}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"
EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-$TOKENIZER_PATH}"
# train script runs from trainer/, need ../ prefix; eval runs from project root
SPLIT_MANIFEST_TRAIN_PATH="${SPLIT_MANIFEST_TRAIN_PATH:-../data_splits/fineedu_qwen3_20b_random_eval_0.01_seed20260602.json}"
SPLIT_MANIFEST_EVAL_PATH="${SPLIT_MANIFEST_EVAL_PATH:-data_splits/fineedu_qwen3_20b_random_eval_0.01_seed20260602.json}"

# ---- 命名空间 & 输出路径 ----
WEIGHT_NAMESPACE="${WEIGHT_NAMESPACE:-fineedu-qwen3-0.6b}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/$WEIGHT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/$WEIGHT_NAMESPACE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fineedu-qwen3-0.6b/$RUN_ID}"
RUN_LOG="$LOG_ROOT/run.log"

# ---- Qwen3-0.6B 架构参数 ----
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-28}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-3072}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
ROPE_THETA="${ROPE_THETA:-1000000}"
RMS_NORM_EPS="${RMS_NORM_EPS:-1e-6}"

# ---- 训练超参 ----
RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"

# ---- Eval ----
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_MODE="${EVAL_MODE:-shards}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-80}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
TAIL_RATIO="${TAIL_RATIO:-0.01}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "[fineedu-qwen3] start at $(date '+%F %T')"
echo "  ROOT               = $ROOT"
echo "  GPUS               = $GPUS"
echo "  SEEDS              = $SEEDS"
echo "  VARIANTS           = $VARIANTS"
echo "  WEIGHT_NAMESPACE   = $WEIGHT_NAMESPACE"
echo "  LOG_ROOT           = $LOG_ROOT"
echo "  FINAL_ROOT         = $FINAL_ROOT"
echo "  RESUME_ROOT        = $RESUME_ROOT"
echo "  DATA_PATH          = $DATA_PATH"
echo "  EVAL_DATA_PATH     = $EVAL_DATA_PATH"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  SPLIT_MANIFEST(train) = $SPLIT_MANIFEST_TRAIN_PATH"
echo "  SPLIT_MANIFEST(eval)  = $SPLIT_MANIFEST_EVAL_PATH"
echo "  HIDDEN_SIZE        = $HIDDEN_SIZE"
echo "  NUM_HIDDEN_LAYERS  = $NUM_HIDDEN_LAYERS"
echo "  NUM_ATTENTION_HEADS= $NUM_ATTENTION_HEADS"
echo "  NUM_KEY_VALUE_HEADS= $NUM_KEY_VALUE_HEADS"
echo "  HEAD_DIM           = $HEAD_DIM"
echo "  INTERMEDIATE_SIZE  = $INTERMEDIATE_SIZE"
echo "  MAX_POSITION_EMB   = $MAX_POSITION_EMBEDDINGS"
echo "  ROPE_THETA         = $ROPE_THETA"
echo "  RMS_NORM_EPS       = $RMS_NORM_EPS"
echo "  LM_HEAD_BIAS       = $LM_HEAD_BIAS"
echo "  RANK               = $RANK"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  EPOCHS             = $EPOCHS"
echo "  SAVE_INTERVAL      = $SAVE_INTERVAL"
echo "  RUN_EVAL           = $RUN_EVAL"
echo "  EVAL_MODE          = $EVAL_MODE"
echo "================================================================"

for seed in $SEEDS; do
    seed_save_dir="$FINAL_ROOT/seed${seed}"
    seed_resume_dir="$RESUME_ROOT/seed${seed}"
    seed_log_dir="$LOG_ROOT/seed${seed}"
    mkdir -p "$seed_log_dir" "$seed_save_dir" "$seed_resume_dir"

    echo
    echo "----------------------------------------------------------------"
    echo "[fineedu-qwen3] seed=$seed"
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
    SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_TRAIN_PATH" \
    HIDDEN_SIZE="$HIDDEN_SIZE" \
    NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
    NUM_ATTENTION_HEADS="$NUM_ATTENTION_HEADS" \
    NUM_KEY_VALUE_HEADS="$NUM_KEY_VALUE_HEADS" \
    HEAD_DIM="$HEAD_DIM" \
    INTERMEDIATE_SIZE="$INTERMEDIATE_SIZE" \
    MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS" \
    ROPE_THETA="$ROPE_THETA" \
    RMS_NORM_EPS="$RMS_NORM_EPS" \
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
    bash "$ROOT/scripts/train/train_large_pretrain.sh"

    if [[ "$RUN_EVAL" == "1" ]]; then
        echo
        echo "[fineedu-qwen3] eval seed=$seed at $(date '+%F %T')"
        PY="$PY" \
        GPU_LIST="$GPUS" \
        EVAL_MODE="$EVAL_MODE" \
        VARIANTS="$VARIANTS" \
        WEIGHT_PREFIX="" \
        SAVE_DIR="$seed_save_dir" \
        DATA_PATH="$EVAL_DATA_PATH" \
        TOKENIZER_PATH="$EVAL_TOKENIZER_PATH" \
        OUTPUT_DIR="$seed_log_dir" \
        SUMMARY_DIR="$LOG_ROOT" \
        SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_EVAL_PATH" \
        TAIL_RATIO="$TAIL_RATIO" \
        HIDDEN_SIZE="$HIDDEN_SIZE" \
        NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
        NUM_ATTENTION_HEADS="$NUM_ATTENTION_HEADS" \
        NUM_KEY_VALUE_HEADS="$NUM_KEY_VALUE_HEADS" \
        HEAD_DIM="$HEAD_DIM" \
        INTERMEDIATE_SIZE="$INTERMEDIATE_SIZE" \
        MAX_POSITION_EMBEDDINGS="$MAX_POSITION_EMBEDDINGS" \
        ROPE_THETA="$ROPE_THETA" \
        RMS_NORM_EPS="$RMS_NORM_EPS" \
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
echo "[fineedu-qwen3] done at $(date '+%F %T')"
echo "  log_root=$LOG_ROOT"
echo "================================================================"
