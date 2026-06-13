#!/usr/bin/env bash
# ==============================================================================
# MiniMind experiment entrypoint with Qwen3-0.6B-sized dense config.
#
# Qwen/Qwen3-0.6B config reference:
#   hidden_size=1024, num_hidden_layers=28, intermediate_size=3072
#   num_attention_heads=16, num_key_value_heads=8, head_dim=128
#   max_position_embeddings=40960, rope_theta=1000000, rms_norm_eps=1e-6
#
# Outputs:
#   weights/final/minimind-qwen3-0.6b-config-tail/seed42/s1_1024.pth
#   weights/resume/minimind-qwen3-0.6b-config-tail/seed42/s1_1024_resume.pth
#   logs/minimind-qwen3-0.6b-config/<run_id>/seed42/s1.log
#
# Examples:
#   bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
#   SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
#   RUN_EVAL=0 MAX_STEPS=10 BATCH_SIZE=16 bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
PY="$TORCH24_PREFIX/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "[qwen3-size-exp] 找不到 torch24 Python: $PY"
    exit 1
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-42 123 2026}"
VARIANTS="${VARIANTS:-s1,s2,s3,s4,s6,s12}"
DATA_PATH="${DATA_PATH:-../dataset/minimind/pretrain_t2t.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-dataset/minimind/pretrain_t2t.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"
EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-$TOKENIZER_PATH}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-}"
TAIL_RATIO="${TAIL_RATIO:-0.01}"

if [[ -n "$SPLIT_MANIFEST_PATH" ]]; then
    SPLIT_MODE="fixed-random-manifest"
    DEFAULT_WEIGHT_NAMESPACE="minimind-qwen3-0.6b-config-randsplit"
else
    SPLIT_MODE="tail-split"
    DEFAULT_WEIGHT_NAMESPACE="minimind-qwen3-0.6b-config-tail"
fi

if [[ -z "${WEIGHT_NAMESPACE+x}" || -z "$WEIGHT_NAMESPACE" ]]; then
    WEIGHT_NAMESPACE="$DEFAULT_WEIGHT_NAMESPACE"
    WEIGHT_NAMESPACE_SOURCE="auto"
else
    WEIGHT_NAMESPACE_SOURCE="user"
fi

FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/$WEIGHT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/$WEIGHT_NAMESPACE}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/minimind-qwen3-0.6b-config/$RUN_ID}"
RUN_LOG="$LOG_ROOT/run.log"

HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-28}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-3072}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-40960}"
ROPE_THETA="${ROPE_THETA:-1000000}"
RMS_NORM_EPS="${RMS_NORM_EPS:-1e-6}"

RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-48}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-0}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "================================================================"
echo "[qwen3-size-exp] start at $(date '+%F %T')"
echo "  ROOT               = $ROOT"
echo "  GPUS               = $GPUS"
echo "  SEEDS              = $SEEDS"
echo "  VARIANTS           = $VARIANTS"
echo "  WEIGHT_NAMESPACE   = $WEIGHT_NAMESPACE"
echo "  NAMESPACE_SOURCE   = $WEIGHT_NAMESPACE_SOURCE"
echo "  SPLIT_MODE         = $SPLIT_MODE"
echo "  LOG_ROOT           = $LOG_ROOT"
echo "  FINAL_ROOT         = $FINAL_ROOT"
echo "  RESUME_ROOT        = $RESUME_ROOT"
echo "  DATA_PATH          = $DATA_PATH"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  SPLIT_MANIFEST     = ${SPLIT_MANIFEST_PATH:-none}"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  TAIL_RATIO         = $TAIL_RATIO"
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
    echo "[qwen3-size-exp] seed=$seed"
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
        echo "[qwen3-size-exp] eval seed=$seed at $(date '+%F %T')"
        CUDA_VISIBLE_DEVICES="$GPUS" "$PY" results/eval_pretrain_loss.py \
            --variants "$VARIANTS" \
            --weight_prefix "" \
            --save_dir "$seed_save_dir" \
            --data_path "$EVAL_DATA_PATH" \
            --tokenizer_path "$EVAL_TOKENIZER_PATH" \
            --hidden_size "$HIDDEN_SIZE" \
            --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
            --num_attention_heads "$NUM_ATTENTION_HEADS" \
            --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
            --head_dim "$HEAD_DIM" \
            --intermediate_size "$INTERMEDIATE_SIZE" \
            --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
            --rope_theta "$ROPE_THETA" \
            --rms_norm_eps "$RMS_NORM_EPS" \
            --tail_ratio "$TAIL_RATIO" \
            --split_manifest_path "$SPLIT_MANIFEST_PATH" \
            --max_samples 0 \
            --batch_size "$EVAL_BATCH_SIZE" \
            --num_workers "$EVAL_NUM_WORKERS" \
            --lm_head_bias "$LM_HEAD_BIAS" \
            --device "$EVAL_DEVICE" \
            --output_csv "$seed_log_dir/eval_pretrain_loss.csv" \
            --output_json "$seed_log_dir/eval_pretrain_loss.json"
    fi
done

echo
echo "================================================================"
echo "[qwen3-size-exp] done at $(date '+%F %T')"
echo "  log_root=$LOG_ROOT"
echo "================================================================"
