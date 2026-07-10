#!/usr/bin/env bash
# FineEdu Qwen3-0.6B -> smol-smoltalk SFT runner.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/..")"
cd "$ROOT"

TORCH24_PREFIX="${TORCH24_PREFIX:-/home/wz/anaconda3/envs/torch24}"
PY="${PY:-$TORCH24_PREFIX/bin/python}"
if [[ ! -x "$PY" ]]; then
    echo "[sft] missing Python: $PY"
    exit 1
fi

export CONDA_PREFIX="$TORCH24_PREFIX"
export CONDA_DEFAULT_ENV="torch24"
export CUDA_HOME="$TORCH24_PREFIX"
export CUDA_PATH="$TORCH24_PREFIX"
export PATH="$TORCH24_PREFIX/bin:${PATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$ROOT/sft/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/sft/.hf_datasets_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
SEEDS="${SEEDS:-42}"
VARIANTS="${VARIANTS:-s1,s3,s12}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$ROOT/sft/smol-smoltalk}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$TRAIN_DATA_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-0.6B}"
INIT_WEIGHT_ROOT="${INIT_WEIGHT_ROOT:-$ROOT/weights/final/fineedu-qwen3-0.6b}"
SFT_NAMESPACE="${SFT_NAMESPACE:-smol-smoltalk-fineedu-qwen3-0.6b}"
FINAL_ROOT="${FINAL_ROOT:-$ROOT/sft/weights/final/$SFT_NAMESPACE}"
RESUME_ROOT="${RESUME_ROOT:-$ROOT/sft/weights/resume/$SFT_NAMESPACE}"
LOG_ROOT="${LOG_ROOT:-$ROOT/sft/logs/$SFT_NAMESPACE/$RUN_ID}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/sft/results/$SFT_NAMESPACE/$RUN_ID}"

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
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
EPOCHS="${EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
SAVE_FINAL="${SAVE_FINAL:-1}"
USE_ATTENTION_MASK="${USE_ATTENTION_MASK:-0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
RUN_SEED_EVAL="${RUN_SEED_EVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FROM_RESUME="${FROM_RESUME:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
DTYPE="${DTYPE:-bfloat16}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

mkdir -p "$LOG_ROOT" "$RESULT_ROOT"
RUN_LOG="$LOG_ROOT/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

export CUDA_VISIBLE_DEVICES="$GPUS"

IFS=',' read -r -a selected_variants <<< "$VARIANTS"

echo "================================================================"
echo "[sft] start at $(date '+%F %T')"
echo "  ROOT             = $ROOT"
echo "  GPUS             = $GPUS (nproc=$NPROC)"
echo "  SEEDS            = $SEEDS"
echo "  VARIANTS         = ${selected_variants[*]}"
echo "  TRAIN_DATA_PATH  = $TRAIN_DATA_PATH"
echo "  TOKENIZER_PATH   = $TOKENIZER_PATH"
echo "  INIT_WEIGHT_ROOT = $INIT_WEIGHT_ROOT"
echo "  FINAL_ROOT       = $FINAL_ROOT"
echo "  RESUME_ROOT      = $RESUME_ROOT"
echo "  LOG_ROOT         = $LOG_ROOT"
echo "  RESULT_ROOT      = $RESULT_ROOT"
echo "  BATCH_SIZE       = $BATCH_SIZE"
echo "  effective_batch  = $((NPROC * BATCH_SIZE * ACCUMULATION_STEPS)) sequences"
echo "  MAX_SEQ_LEN      = $MAX_SEQ_LEN"
echo "  EPOCHS           = $EPOCHS"
echo "  MAX_STEPS        = $MAX_STEPS"
echo "  RUN_SEED_EVAL    = $RUN_SEED_EVAL"
echo "================================================================"

for seed in $SEEDS; do
    seed_final_dir="$FINAL_ROOT/seed${seed}"
    seed_resume_dir="$RESUME_ROOT/seed${seed}"
    seed_log_dir="$LOG_ROOT/seed${seed}"
    seed_result_dir="$RESULT_ROOT/seed${seed}"
    mkdir -p "$seed_final_dir" "$seed_resume_dir" "$seed_log_dir" "$seed_result_dir"

    for variant in "${selected_variants[@]}"; do
        init_weight="$INIT_WEIGHT_ROOT/seed${seed}/${variant}_${HIDDEN_SIZE}.pth"
        final_weight="$seed_final_dir/${variant}_${HIDDEN_SIZE}.pth"
        logfile="$seed_log_dir/${variant}.log"
        metrics_path="$seed_result_dir/${variant}_metrics.jsonl"

        echo
        echo "----------------------------------------------------------------"
        echo "[sft] seed=$seed variant=$variant"
        echo "  init_weight=$init_weight"
        echo "  final_weight=$final_weight"
        echo "  log=$logfile"
        echo "----------------------------------------------------------------"

        if [[ ! -f "$init_weight" ]]; then
            echo "[sft] missing init weight: $init_weight"
            exit 1
        fi
        if [[ "$SKIP_COMPLETED" == "1" && -f "$final_weight" ]]; then
            echo "[sft] skip completed: $final_weight"
            continue
        fi

        "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$ROOT/sft/train_sft.py" \
            --train_data_path "$TRAIN_DATA_PATH" \
            --eval_data_path "$EVAL_DATA_PATH" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --init_weight_path "$init_weight" \
            --output_dir "$seed_final_dir" \
            --checkpoint_dir "$seed_resume_dir" \
            --save_weight "$variant" \
            --embedding_variant "$variant" \
            --embedding_variant_rank "$RANK" \
            --seed "$seed" \
            --hidden_size "$HIDDEN_SIZE" \
            --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
            --num_attention_heads "$NUM_ATTENTION_HEADS" \
            --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
            --head_dim "$HEAD_DIM" \
            --intermediate_size "$INTERMEDIATE_SIZE" \
            --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
            --rope_theta "$ROPE_THETA" \
            --rms_norm_eps "$RMS_NORM_EPS" \
            --lm_head_bias "$LM_HEAD_BIAS" \
            --batch_size "$BATCH_SIZE" \
            --accumulation_steps "$ACCUMULATION_STEPS" \
            --max_seq_len "$MAX_SEQ_LEN" \
            --epochs "$EPOCHS" \
            --learning_rate "$LEARNING_RATE" \
            --log_interval "$LOG_INTERVAL" \
            --save_interval "$SAVE_INTERVAL" \
            --save_final "$SAVE_FINAL" \
            --use_attention_mask "$USE_ATTENTION_MASK" \
            --eval_interval "$EVAL_INTERVAL" \
            --eval_max_batches "$EVAL_MAX_BATCHES" \
            --num_workers "$NUM_WORKERS" \
            --from_resume "$FROM_RESUME" \
            --metrics_path "$metrics_path" \
            --max_steps "$MAX_STEPS" \
            --max_train_samples "$MAX_TRAIN_SAMPLES" \
            --max_eval_samples "$MAX_EVAL_SAMPLES" \
            --dtype "$DTYPE" \
            2>&1 | tee "$logfile"
    done

    if [[ "$RUN_SEED_EVAL" == "1" ]]; then
        seed_eval_log_dir="$seed_log_dir/eval_after_seed"
        seed_eval_result_dir="$seed_result_dir/eval_after_seed"
        seed_eval_summary="$seed_eval_result_dir/seed${seed}_eval_summary.jsonl"
        mkdir -p "$seed_eval_log_dir" "$seed_eval_result_dir"
        rm -f "$seed_eval_summary"

        echo
        echo "----------------------------------------------------------------"
        echo "[sft-eval] seed=$seed variants=${selected_variants[*]}"
        echo "  summary=$seed_eval_summary"
        echo "----------------------------------------------------------------"

        for variant in "${selected_variants[@]}"; do
            final_weight="$seed_final_dir/${variant}_${HIDDEN_SIZE}.pth"
            eval_logfile="$seed_eval_log_dir/${variant}_eval.log"

            if [[ ! -f "$final_weight" ]]; then
                echo "[sft-eval] missing final weight: $final_weight"
                exit 1
            fi

            "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$ROOT/sft/eval_sft.py" \
                --eval_data_path "$EVAL_DATA_PATH" \
                --tokenizer_path "$TOKENIZER_PATH" \
                --weight_path "$final_weight" \
                --metrics_path "$seed_eval_summary" \
                --run_tag "after_seed" \
                --embedding_variant "$variant" \
                --embedding_variant_rank "$RANK" \
                --seed "$seed" \
                --hidden_size "$HIDDEN_SIZE" \
                --num_hidden_layers "$NUM_HIDDEN_LAYERS" \
                --num_attention_heads "$NUM_ATTENTION_HEADS" \
                --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
                --head_dim "$HEAD_DIM" \
                --intermediate_size "$INTERMEDIATE_SIZE" \
                --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
                --rope_theta "$ROPE_THETA" \
                --rms_norm_eps "$RMS_NORM_EPS" \
                --lm_head_bias "$LM_HEAD_BIAS" \
                --batch_size "$BATCH_SIZE" \
                --max_seq_len "$MAX_SEQ_LEN" \
                --eval_max_batches "$EVAL_MAX_BATCHES" \
                --max_eval_samples "$MAX_EVAL_SAMPLES" \
                --num_workers "$NUM_WORKERS" \
                --dtype "$DTYPE" \
                --use_attention_mask "$USE_ATTENTION_MASK" \
                2>&1 | tee "$eval_logfile"
        done
    fi
done

EVAL_SUMMARY_CSV=""
if [[ "$RUN_SEED_EVAL" == "1" ]]; then
    EVAL_SUMMARY_CSV="$RESULT_ROOT/eval_summary.csv"
    "$PY" "$ROOT/sft/summarize_results.py" \
        --result-root "$RESULT_ROOT" \
        --output "$EVAL_SUMMARY_CSV" \
        --seeds "$SEEDS" \
        --variants "$VARIANTS"
fi

echo
echo "================================================================"
echo "[sft] done at $(date '+%F %T')"
echo "  final_root=$FINAL_ROOT"
echo "  log_root=$LOG_ROOT"
echo "  result_root=$RESULT_ROOT"
if [[ -n "$EVAL_SUMMARY_CSV" ]]; then
    echo "  eval_summary=$EVAL_SUMMARY_CSV"
fi
echo "================================================================"
