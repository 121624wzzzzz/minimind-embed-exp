#!/usr/bin/env bash
# ==============================================================================
# FineWeb-Edu GPT-2-tokenizer pretrain runner.
#
# Defaults target the 6B-token packed FineWeb-Edu dataset and the ten variants
# used by both the historical tail and fixed-random experiments.
#
# Examples:
#   bash scripts/train/train_fineedu_gpt2_pretrain.sh
#   MAX_STEPS=5 DATA_PATH=dataset/fineweb_edu/smoke bash scripts/train/train_fineedu_gpt2_pretrain.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="${TORCH24_PREFIX:-/home/wz/anaconda3/envs/torch24}"
PY="${PY:-$TORCH24_PREFIX/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[fineedu] 找不到 Python: $PY"
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

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATA_PATH="${DATA_PATH:-$ROOT/dataset/fineweb_edu/packed/gpt2_6b_seq340}"
TOKENIZER_PATH="${TOKENIZER_PATH:-gpt2}"
SAVE_DIR="${SAVE_DIR:-$ROOT/weights/final}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT/weights/resume}"
WEIGHT_PREFIX="${WEIGHT_PREFIX-fineedu_gpt2_6b}"
SEED="${SEED:-42}"
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
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-}"
GRAD_LOG_INTERVAL="${GRAD_LOG_INTERVAL:-1000}"
GRAD_SAVE_TENSORS="${GRAD_SAVE_TENSORS:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/fineedu-gpt2-6b-seed${SEED}}"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

if [[ -n "${VARIANTS:-}" ]]; then
    IFS=',' read -r -a selected_variants <<< "$VARIANTS"
else
    selected_variants=(s1 s2 s3 s4 s5 s6 s7 s11 s12 s13)
fi

echo "================================================================"
echo "[fineedu] start at $(date '+%F %T')"
echo "  GPUS               = $GPUS  (DDP nproc=$NPROC)"
echo "  VARIANTS           = ${selected_variants[*]}"
echo "  DATA_PATH          = $DATA_PATH"
echo "  TOKENIZER_PATH     = $TOKENIZER_PATH"
echo "  WEIGHT_PREFIX      = $WEIGHT_PREFIX"
echo "  SEED               = $SEED"
echo "  RANK               = $RANK"
echo "  LM_HEAD_BIAS       = $LM_HEAD_BIAS"
echo "  TRAIN_SPLIT_RATIO  = $TRAIN_SPLIT_RATIO"
echo "  SPLIT_MANIFEST     = ${SPLIT_MANIFEST_PATH:-none}"
echo "  BATCH_SIZE         = $BATCH_SIZE"
echo "  ACCUMULATION_STEPS = $ACCUMULATION_STEPS"
echo "  effective_batch    = $((NPROC * BATCH_SIZE * ACCUMULATION_STEPS)) sequences"
echo "  MAX_SEQ_LEN        = $MAX_SEQ_LEN"
echo "  EPOCHS             = $EPOCHS"
echo "  LEARNING_RATE      = $LEARNING_RATE"
echo "  GRAD_LOG_INTERVAL  = $GRAD_LOG_INTERVAL"
echo "  GRAD_SAVE_TENSORS  = $GRAD_SAVE_TENSORS"
echo "  SKIP_COMPLETED     = $SKIP_COMPLETED"
echo "  FROM_RESUME        = $FROM_RESUME"
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
    local grad_log="$LOG_DIR/${variant}_grad_stats.jsonl"
    local final_weight="$SAVE_DIR/${save_weight}_768.pth"
    local started=$(date +%s)

    echo
    echo "----------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] >>> START variant: $variant"
    echo "  save_weight: $save_weight"
    echo "  log: $logfile"
    echo "  grad_log: $grad_log"
    echo "----------------------------------------------------------------"

    if [[ "$SKIP_COMPLETED" == "1" ]]; then
        if [[ -f "$final_weight" ]] && \
            { grep -Eq "<<< (DONE|FINISH) $variant" "$logfile" 2>/dev/null || \
                tail -n 20 "$logfile" 2>/dev/null | grep -Pq "Epoch:\[$EPOCHS/$EPOCHS\]\(([0-9]+)/\1\)"; }; then
            echo "[$(date '+%H:%M:%S')] --- SKIP completed variant: $variant"
            return 0
        fi
    fi

    if "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$ROOT/trainer/train_pretrain.py" \
        --data_path "$DATA_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --save_dir "$SAVE_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --save_weight "$save_weight" \
        --embedding_variant "$variant" \
        --embedding_variant_rank "$RANK" \
        --seed "$SEED" \
        --lm_head_bias "$LM_HEAD_BIAS" \
        --train_split_ratio "$TRAIN_SPLIT_RATIO" \
        --split_manifest_path "$SPLIT_MANIFEST_PATH" \
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
        --grad_log_interval "$GRAD_LOG_INTERVAL" \
        --grad_log_path "$grad_log" \
        --grad_save_tensors "$GRAD_SAVE_TENSORS" \
        --grad_tensor_dir "$LOG_DIR/${variant}_grad_tensors" \
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
echo "[fineedu] ALL DONE at $(date '+%F %T')"
echo "权重产出:"
if [[ -n "$WEIGHT_PREFIX" ]]; then
    ls -lh "$SAVE_DIR"/"${WEIGHT_PREFIX}"_s*_768.pth 2>/dev/null || echo "  (无)"
else
    ls -lh "$SAVE_DIR"/s*_768.pth 2>/dev/null || echo "  (无)"
fi
echo "================================================================"
