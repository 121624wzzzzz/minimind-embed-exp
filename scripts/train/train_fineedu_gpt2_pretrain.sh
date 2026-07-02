#!/usr/bin/env bash
# ==============================================================================
# FineWeb-Edu GPT-2-tokenizer pretrain runner for key S variants.
#
# Defaults target the 6B-token packed FineWeb-Edu dataset and run only:
#   S1, S2, S3, S6, S12
#
# Examples:
#   bash scripts/train/train_fineedu_gpt2_pretrain.sh
#   MAX_STEPS=5 DATA_PATH=../dataset/fineweb_edu/smoke bash scripts/train/train_fineedu_gpt2_pretrain.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

TORCH24_PREFIX="/home/wz/anaconda3/envs/torch24"
if [[ ! -x "$TORCH24_PREFIX/bin/python" ]]; then
    echo "[fineedu] 找不到 torch24 Python: $TORCH24_PREFIX/bin/python"
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
DATA_PATH="${DATA_PATH:-../dataset/fineweb_edu/packed/gpt2_6b_seq340}"
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
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_TAIL_RATIO="${EVAL_TAIL_RATIO:-0.01}"

mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

if [[ -n "${VARIANTS:-}" ]]; then
    IFS=',' read -r -a selected_variants <<< "$VARIANTS"
else
    selected_variants=(s1 s2 s3 s6 s12)
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
            { grep -q "<<< FINISH $variant" "$logfile" 2>/dev/null || \
                tail -n 20 "$logfile" 2>/dev/null | grep -Pq "Epoch:\[$EPOCHS/$EPOCHS\]\(([0-9]+)/\1\)"; }; then
            echo "[$(date '+%H:%M:%S')] --- SKIP completed variant: $variant"
            return 0
        fi
    fi

    cd "$ROOT/trainer"
    if torchrun --standalone --nproc_per_node="$NPROC" train_pretrain.py \
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
    cd "$ROOT"
}

for variant in "${selected_variants[@]}"; do
    run_variant "$variant"
done

if [[ "$RUN_EVAL" == "1" ]]; then
    cd "$ROOT"

    eval_tail_ratio="$EVAL_TAIL_RATIO"
    if [[ -n "$SPLIT_MANIFEST_PATH" ]]; then
        eval_tail_ratio="0"
    fi

    IFS=',' read -r -a gpu_ids <<< "$GPUS"
    eval_variants_arr=("${selected_variants[@]}")
    num_gpus=${#gpu_ids[@]}
    num_variants=${#eval_variants_arr[@]}

    # Distribute variants round-robin across GPUs
    gpu_variant_groups=()
    for ((i = 0; i < num_variants; i++)); do
        gpu_idx=$((i % num_gpus))
        if [[ -n "${gpu_variant_groups[$gpu_idx]:-}" ]]; then
            gpu_variant_groups[$gpu_idx]="${gpu_variant_groups[$gpu_idx]},${eval_variants_arr[$i]}"
        else
            gpu_variant_groups[$gpu_idx]="${eval_variants_arr[$i]}"
        fi
    done

    eval_tmpdir="$LOG_DIR/eval_tmp_$$"
    mkdir -p "$eval_tmpdir"
    trap 'rm -rf "$eval_tmpdir"' EXIT
    eval_pids=()

    for ((gpu_idx = 0; gpu_idx < num_gpus; gpu_idx++)); do
        group_variants="${gpu_variant_groups[$gpu_idx]:-}"
        [[ -z "$group_variants" ]] && continue
        gpu_id="${gpu_ids[$gpu_idx]}"
        tmp_csv="$eval_tmpdir/gpu${gpu_id}.csv"
        tmp_json="$eval_tmpdir/gpu${gpu_id}.json"

        echo "[eval] launching GPU $gpu_id => variants: $group_variants"
        env CUDA_VISIBLE_DEVICES="$gpu_id" "$TORCH24_PREFIX/bin/python" results/eval_pretrain_loss.py \
            --variants "$group_variants" \
            --weight_prefix "$WEIGHT_PREFIX" \
            --save_dir "$SAVE_DIR" \
            --data_path "${DATA_PATH#../}" \
            --tokenizer_path "$TOKENIZER_PATH" \
            --tail_ratio "$eval_tail_ratio" \
            --split_manifest_path "$SPLIT_MANIFEST_PATH" \
            --max_samples 0 \
            --batch_size "$EVAL_BATCH_SIZE" \
            --num_workers 2 \
            --lm_head_bias "$LM_HEAD_BIAS" \
            --device "cuda:0" \
            --output_csv "$tmp_csv" \
            --output_json "$tmp_json" &
        eval_pids+=($!)
    done

    eval_failed=0
    for pid in "${eval_pids[@]}"; do
        if ! wait "$pid"; then
            echo "[eval] GPU eval process $pid failed!"
            eval_failed=1
        fi
    done

    if [[ "$eval_failed" == "1" ]]; then
        echo "[eval] some eval shards failed, aborting"
        exit 1
    fi

    # Merge shards and recompute deltas against the global S1/S2 results.
    # A shard cannot calculate a baseline delta when that baseline ran on another GPU.
    expected_variants="$(IFS=','; echo "${eval_variants_arr[*]}")"
    "$TORCH24_PREFIX/bin/python" - \
        "$eval_tmpdir" \
        "$LOG_DIR/eval_pretrain_loss.csv" \
        "$LOG_DIR/eval_pretrain_loss.json" \
        "$expected_variants" <<'PY'
import csv
import glob
import json
import os
import sys
from decimal import Decimal

tmpdir, csv_path, json_path, expected_csv = sys.argv[1:]
expected = expected_csv.split(",")
fields = [
    "variant", "loss", "ppl", "delta_vs_s1", "delta_vs_s2",
    "tokens", "sequences", "seconds", "data_path", "start_index", "max_samples",
]

rows = []
for path in sorted(glob.glob(os.path.join(tmpdir, "gpu*.json"))):
    with open(path, encoding="utf-8") as fh:
        rows.extend(json.load(fh))

by_variant = {row["variant"]: row for row in rows}
if len(by_variant) != len(rows):
    raise RuntimeError("duplicate variants found while merging eval shards")
missing = [variant for variant in expected if variant not in by_variant]
unexpected = sorted(set(by_variant) - set(expected))
if missing or unexpected:
    raise RuntimeError(f"incomplete eval shards: missing={missing}, unexpected={unexpected}")

rows = [by_variant[variant] for variant in expected]
baselines = {
    "delta_vs_s1": Decimal(by_variant["s1"]["loss"]) if "s1" in by_variant else None,
    "delta_vs_s2": Decimal(by_variant["s2"]["loss"]) if "s2" in by_variant else None,
}
for row in rows:
    loss = Decimal(row["loss"])
    for field, baseline in baselines.items():
        row[field] = "" if baseline is None else f"{loss - baseline:+.6f}"

with open(csv_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
with open(json_path, "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=2)
print(f"[eval] merged {len(rows)} variants -> {csv_path}")
print(f"[eval] merged JSON -> {json_path}")
PY
    rm -rf "$eval_tmpdir"
    trap - EXIT
fi

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
