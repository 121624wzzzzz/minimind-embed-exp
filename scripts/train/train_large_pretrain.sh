#!/usr/bin/env bash
# ==============================================================================
# MiniMind-Large pretrain runner (d=1024, L=16, ~150M params).
# Core variants: S1, S2, S3, S6, S12
#
# Usage:
#   bash scripts/train/train_large_pretrain.sh                                     # 正式跑
#   MAX_STEPS=10 BATCH_SIZE=48 bash scripts/train/train_large_pretrain.sh           # smoke test
#   VARIANTS=s1,s3 SEED=42 bash scripts/train/train_large_pretrain.sh               # 单变体
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="${TORCH24_PREFIX:-/home/wz/anaconda3/envs/torch24}"
PY="${PY:-$TORCH24_PREFIX/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[large] 找不到 Python: $PY"
    exit 1
fi

export CONDA_PREFIX="$TORCH24_PREFIX"
export CONDA_DEFAULT_ENV="torch24"
export CUDA_HOME="$TORCH24_PREFIX"
export CUDA_PATH="$TORCH24_PREFIX"
export PATH="$TORCH24_PREFIX/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

# ---- 模型架构 ----
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
HIDDEN_SIZE="${HIDDEN_SIZE:-1024}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-16}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-0}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-0}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-32768}"
ROPE_THETA="${ROPE_THETA:-1000000}"
RMS_NORM_EPS="${RMS_NORM_EPS:-1e-6}"

# ---- 训练参数 ----
DATA_PATH="${DATA_PATH:-$ROOT/dataset/minimind/pretrain_t2t.jsonl}"
SAVE_DIR="${SAVE_DIR:-$ROOT/weights/final}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/weights/resume}"
SEED="${SEED:-42}"
WEIGHT_PREFIX="${WEIGHT_PREFIX-pretrain_large_seed${SEED}}"
RANK="${RANK:-32}"
BATCH_SIZE="${BATCH_SIZE:-168}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
TRAIN_SPLIT_RATIO="${TRAIN_SPLIT_RATIO:-0.99}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/model}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/minimind/large_seed${SEED}}"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

# ---- 核心变体 ----
all_variants=(s1 s2 s3 s4 s6 s12)
selected_variants=()
if [[ -n "${VARIANTS:-}" ]]; then
    IFS=',' read -r -a selected_variants <<< "$VARIANTS"
else
    selected_variants=("${all_variants[@]}")
fi

echo "================================================================"
echo "[large] start at $(date '+%F %T')"
echo "  GPUS               = $GPUS  (DDP nproc=$NPROC)"
echo "  HIDDEN_SIZE        = $HIDDEN_SIZE"
echo "  NUM_HIDDEN_LAYERS  = $NUM_HIDDEN_LAYERS"
echo "  NUM_ATTENTION_HEADS= $NUM_ATTENTION_HEADS"
echo "  NUM_KEY_VALUE_HEADS= $NUM_KEY_VALUE_HEADS"
echo "  HEAD_DIM           = $HEAD_DIM"
echo "  INTERMEDIATE_SIZE  = $INTERMEDIATE_SIZE"
echo "  MAX_POSITION_EMB   = $MAX_POSITION_EMBEDDINGS"
echo "  ROPE_THETA         = $ROPE_THETA"
echo "  RMS_NORM_EPS       = $RMS_NORM_EPS"
echo "  VARIANTS           = ${selected_variants[*]}"
echo "  DATA_PATH          = $DATA_PATH"
echo "  SAVE_DIR           = $SAVE_DIR"
echo "  CHECKPOINT_DIR     = $CHECKPOINT_DIR"
echo "  WEIGHT_PREFIX      = $WEIGHT_PREFIX"
echo "  SEED               = $SEED"
echo "  RANK               = $RANK"
echo "  LM_HEAD_BIAS       = $LM_HEAD_BIAS"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  SPLIT_MANIFEST     = ${SPLIT_MANIFEST_PATH:-none}"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  GRAD_LOG_INTERVAL  = $GRAD_LOG_INTERVAL"
echo "  GRAD_SAVE_TENSORS  = $GRAD_SAVE_TENSORS"
echo "  SKIP_COMPLETED     = $SKIP_COMPLETED"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  ACCUMULATION_STEPS = $ACCUMULATION_STEPS"
echo "  effective_batch    = $((NPROC * BATCH_SIZE * ACCUMULATION_STEPS)) sequences"
echo "  MAX_SEQ_LEN        = $MAX_SEQ_LEN"
echo "  EPOCHS             = $EPOCHS"
echo "  LEARNING_RATE      = $LEARNING_RATE"
echo "  MAX_STEPS          = $MAX_STEPS"
echo "  LOG_DIR            = $LOG_DIR"
echo "================================================================"

run_variant() {
    local variant=$1
    local save_weight="$variant"
    if [[ -n "$WEIGHT_PREFIX" ]]; then
        save_weight="${WEIGHT_PREFIX}_${variant}"
    fi
    local logfile="$LOG_DIR/${variant}.log"
    local final_weight="$SAVE_DIR/${save_weight}_${HIDDEN_SIZE}.pth"
    local started=$(date +%s)

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] >>> START $variant"
    echo "  save_weight: $save_weight"
    echo "  log: $logfile"
    echo "----------------------------------------------------------------"

    if [[ "$SKIP_COMPLETED" == "1" && -f "$final_weight" ]]; then
        if grep -Eq "<<< (DONE|FINISH) $variant" "$logfile" 2>/dev/null || \
            tail -n 20 "$logfile" 2>/dev/null | grep -Pq "Epoch:\[$EPOCHS/$EPOCHS\]\(([0-9]+)/\1\)"; then
            echo "[$(date '+%H:%M:%S')] --- SKIP completed variant: $variant"
            return 0
        fi
    fi

    if "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$ROOT/trainer/train_pretrain.py" \
        --hidden_size "$HIDDEN_SIZE" \
        --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
        --num_attention_heads "$NUM_ATTENTION_HEADS" \
        --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
        --head_dim "$HEAD_DIM" \
        --intermediate_size "$INTERMEDIATE_SIZE" \
        --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
        --rope_theta "$ROPE_THETA" \
        --rms_norm_eps "$RMS_NORM_EPS" \
        --data_path "$DATA_PATH" \
        --save_dir "$SAVE_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --save_weight "$save_weight" \
        --embedding_variant "$variant" \
        --embedding_variant_rank "$RANK" \
        --seed "$SEED" \
        --lm_head_bias "$LM_HEAD_BIAS" \
        --train_split_ratio "$TRAIN_SPLIT_RATIO" \
        --split_manifest_path "$SPLIT_MANIFEST_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --grad_log_interval "$GRAD_LOG_INTERVAL" \
        --grad_log_path "$LOG_DIR/${variant}_grad_stats.jsonl" \
        --grad_save_tensors "$GRAD_SAVE_TENSORS" \
        --grad_tensor_dir "$LOG_DIR/${variant}_grad_tensors" \
        --batch_size "$BATCH_SIZE" \
        --accumulation_steps "$ACCUMULATION_STEPS" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --log_interval "$LOG_INTERVAL" \
        --save_interval "$SAVE_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --from_resume "$FROM_RESUME" \
        --max_steps "$MAX_STEPS" \
        2>&1 | tee "$logfile"; then
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] <<< FINISH $variant in $((elapsed/60))m$((elapsed%60))s" | tee -a "$logfile"
    else
        local elapsed=$(( $(date +%s) - started ))
        echo "[$(date '+%H:%M:%S')] !!! FAIL $variant after $((elapsed/60))m$((elapsed%60))s" | tee -a "$logfile"
        exit 1
    fi
}

for variant in "${selected_variants[@]}"; do
    run_variant "$variant"
done

echo
echo "================================================================"
echo "[large] ALL DONE at $(date '+%F %T')"
echo "权重产出:"
save_dir_path="$SAVE_DIR"
if [[ -n "$WEIGHT_PREFIX" ]]; then
    ls -lh "$save_dir_path"/"${WEIGHT_PREFIX}"_s*_"${HIDDEN_SIZE}".pth 2>/dev/null || echo "  (无)"
else
    ls -lh "$save_dir_path"/s*_"${HIDDEN_SIZE}".pth 2>/dev/null || echo "  (无)"
fi
echo "================================================================"
