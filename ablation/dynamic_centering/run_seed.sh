#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="${TORCH24_PREFIX:-/home/wz/anaconda3/envs/torch24}"
PY="${PY:-$TORCH24_PREFIX/bin/python}"
[[ -x "$PY" ]] || { echo "[dynamic-centering] 找不到 Python: $PY" >&2; exit 1; }

export CONDA_PREFIX="$TORCH24_PREFIX"
export CONDA_DEFAULT_ENV="torch24"
export CUDA_HOME="$TORCH24_PREFIX"
export CUDA_PATH="$TORCH24_PREFIX"
export PATH="$TORCH24_PREFIX/bin:${PATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="$(tr ',' '\n' <<< "$GPUS" | wc -l)"
SEED="${SEED:-42}"
VARIANT=center_dynamic
RUN_ID="${RUN_ID:-seed${SEED}_full_$(date +%Y%m%d_%H%M%S)}"

DATA_PATH="${DATA_PATH:-$ROOT/dataset/minimind/pretrain_t2t.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ROOT/model}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-$ROOT/data_splits/minimind_pretrain_t2t_random_eval_0.01_seed20260602.json}"
BASELINE_CSV="${BASELINE_CSV:-$ROOT/docs/eval_results/minimind-small-fixedinit-randsplit/seed${SEED}/eval_pretrain_loss.csv}"

FINAL_ROOT="${FINAL_ROOT:-$ROOT/weights/final/minimind-small-dynamic-centering-randsplit/seed${SEED}}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/weights/resume/minimind-small-dynamic-centering-randsplit/seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/dynamic-centering/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/minimind-small-fixedinit-randsplit}"
SEED_RESULT_DIR="${SEED_RESULT_DIR:-$RESULT_ROOT/seed${SEED}}"

BATCH_SIZE="${BATCH_SIZE:-224}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MAX_STEPS="${MAX_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
FROM_RESUME="${FROM_RESUME:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE="${OVERWRITE:-0}"

for required in "$DATA_PATH" "$SPLIT_MANIFEST_PATH" "$BASELINE_CSV"; do
    [[ -f "$required" ]] || { echo "[dynamic-centering] 缺少文件: $required" >&2; exit 1; }
done

FINAL_WEIGHT="$FINAL_ROOT/${VARIANT}_768.pth"
if [[ -f "$FINAL_WEIGHT" && "$FROM_RESUME" != "1" && "$OVERWRITE" != "1" ]]; then
    echo "[dynamic-centering] 权重已存在，拒绝覆盖: $FINAL_WEIGHT" >&2
    echo "如需重跑，请显式设置 OVERWRITE=1；如需续训，请设置 FROM_RESUME=1。" >&2
    exit 1
fi

mkdir -p "$FINAL_ROOT" "$RESUME_ROOT" "$LOG_ROOT" "$SEED_RESULT_DIR"
exec > >(tee -a "$LOG_ROOT/run.log") 2>&1

echo "================================================================"
echo "[dynamic-centering] start at $(date '+%F %T')"
echo "  RUN_ID             = $RUN_ID"
echo "  GPUS               = $GPUS (nproc=$NPROC)"
echo "  VARIANT            = $VARIANT"
echo "  SEED               = $SEED"
echo "  DATA_PATH          = $DATA_PATH"
echo "  SPLIT_MANIFEST     = $SPLIT_MANIFEST_PATH"
echo "  MODEL              = hidden768/layers8/q8/kv4"
echo "  BATCH_SIZE         = $BATCH_SIZE/GPU"
echo "  effective_batch    = $((NPROC * BATCH_SIZE * ACCUMULATION_STEPS)) sequences"
echo "  MAX_SEQ_LEN        = $MAX_SEQ_LEN"
echo "  EPOCHS             = $EPOCHS"
echo "  LEARNING_RATE      = $LEARNING_RATE"
echo "  MAX_STEPS          = $MAX_STEPS (0 means full budget)"
echo "  FINAL_ROOT         = $FINAL_ROOT"
echo "  RESUME_ROOT        = $RESUME_ROOT"
echo "  LOG_ROOT           = $LOG_ROOT"
echo "  RESULT_ROOT        = $RESULT_ROOT"
echo "  SEED_RESULT_DIR    = $SEED_RESULT_DIR"
echo "================================================================"
sha256sum "$SPLIT_MANIFEST_PATH"

"$PY" "$SCRIPT_DIR/verify_implementation.py" \
    --output "$SEED_RESULT_DIR/implementation_checks.json"

export CUDA_VISIBLE_DEVICES="$GPUS"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    "$ROOT/trainer/train_pretrain.py" \
    --data_path "$DATA_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --split_manifest_path "$SPLIT_MANIFEST_PATH" \
    --save_dir "$FINAL_ROOT" \
    --checkpoint_dir "$RESUME_ROOT" \
    --save_weight "$VARIANT" \
    --embedding_variant "$VARIANT" \
    --embedding_variant_rank 32 \
    --seed "$SEED" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 4 \
    --head_dim 0 \
    --intermediate_size 0 \
    --max_position_embeddings 32768 \
    --rope_theta 1000000 \
    --rms_norm_eps 1e-6 \
    --tie_word_embeddings 1 \
    --lm_head_bias 1 \
    --use_moe 0 \
    --dtype bfloat16 \
    --batch_size "$BATCH_SIZE" \
    --accumulation_steps "$ACCUMULATION_STEPS" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --num_workers "$NUM_WORKERS" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --from_resume "$FROM_RESUME" \
    --max_steps "$MAX_STEPS"

if [[ "$RUN_EVAL" == "1" ]]; then
    GPU_LIST="$GPUS" \
    EVAL_MODE=shards \
    VARIANTS="$VARIANT" \
    WEIGHT_PREFIX="" \
    SAVE_DIR="$FINAL_ROOT" \
    DATA_PATH="$DATA_PATH" \
    TOKENIZER_PATH="$TOKENIZER_PATH" \
    OUTPUT_DIR="$SEED_RESULT_DIR" \
    SPLIT_MANIFEST_PATH="$SPLIT_MANIFEST_PATH" \
    HIDDEN_SIZE=768 \
    NUM_HIDDEN_LAYERS=8 \
    NUM_ATTENTION_HEADS=8 \
    NUM_KEY_VALUE_HEADS=4 \
    EMBEDDING_VARIANT_RANK=32 \
    LM_HEAD_BIAS=1 \
    MAX_SEQ_LEN="$MAX_SEQ_LEN" \
    EVAL_BATCH_SIZE=64 \
    EVAL_NUM_WORKERS=4 \
    EVAL_DTYPE=bfloat16 \
    PY="$PY" \
    bash "$ROOT/scripts/eval/eval_parallel.sh"

    # 通用 evaluator 按 CSV 默认写入 CRLF；正式归档统一为仓库使用的 LF。
    sed -i 's/\r$//' "$SEED_RESULT_DIR/eval_pretrain_loss.csv"

    "$PY" "$SCRIPT_DIR/summarize_results.py" \
        --baseline_csv "$BASELINE_CSV" \
        --center_json "$SEED_RESULT_DIR/eval_pretrain_loss.json" \
        --output_csv "$SEED_RESULT_DIR/comparison.csv" \
        --output_json "$SEED_RESULT_DIR/comparison.json"
fi

echo "================================================================"
echo "[dynamic-centering] done at $(date '+%F %T')"
echo "  final_weight=$FINAL_WEIGHT"
echo "  seed_result_dir=$SEED_RESULT_DIR"
echo "================================================================"
