#!/usr/bin/env bash
# Evaluate one seed across multiple GPUs and merge results.
#
# EVAL_MODE=variants assigns different variants to different GPUs.
# EVAL_MODE=shards evaluates each variant on all GPUs by sharding the data.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

PY="${PY:-python}"
EVALUATOR="${EVALUATOR:-$ROOT/results/eval_pretrain_loss.py}"
GPU_LIST="${GPU_LIST:-${GPUS:-0}}"
EVAL_MODE="${EVAL_MODE:-variants}"
VARIANTS_CSV="${VARIANTS:-s1}"

: "${SAVE_DIR:?SAVE_DIR is required}"
: "${DATA_PATH:?DATA_PATH is required}"
: "${TOKENIZER_PATH:?TOKENIZER_PATH is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

WEIGHT_PREFIX="${WEIGHT_PREFIX:-}"
WORK_ROOT="${WORK_DIR:-$OUTPUT_DIR/eval_parallel}"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTPUT_DIR/eval_pretrain_loss.csv}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/eval_pretrain_loss.json}"
SUMMARY_DIR="${SUMMARY_DIR:-}"

HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
HEAD_DIM="${HEAD_DIM:-0}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-0}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-32768}"
ROPE_THETA="${ROPE_THETA:-1000000}"
RMS_NORM_EPS="${RMS_NORM_EPS:-1e-6}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
EMBEDDING_VARIANT_RANK="${EMBEDDING_VARIANT_RANK:-32}"
LM_HEAD_BIAS="${LM_HEAD_BIAS:-1}"
USE_MOE="${USE_MOE:-0}"

SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-}"
TAIL_RATIO="${TAIL_RATIO:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
START_INDEX="${START_INDEX:-0}"
MAX_BATCHES="${MAX_BATCHES:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

case "$EVAL_MODE" in
    variants|shards) ;;
    *)
        echo "[parallel-eval] EVAL_MODE must be variants or shards, got: $EVAL_MODE" >&2
        exit 2
        ;;
esac

gpu_values="${GPU_LIST//,/ }"
read -r -a GPUS <<< "$gpu_values"
IFS=',' read -r -a RAW_VARIANTS <<< "$VARIANTS_CSV"
VARIANTS=()
for variant in "${RAW_VARIANTS[@]}"; do
    variant="${variant//[[:space:]]/}"
    [[ -n "$variant" ]] && VARIANTS+=("$variant")
done

if (( ${#GPUS[@]} == 0 )); then
    echo "[parallel-eval] GPU_LIST is empty" >&2
    exit 2
fi
if (( ${#VARIANTS[@]} == 0 )); then
    echo "[parallel-eval] VARIANTS is empty" >&2
    exit 2
fi
if [[ ! -f "$EVALUATOR" ]]; then
    echo "[parallel-eval] evaluator not found: $EVALUATOR" >&2
    exit 2
fi

if [[ -n "$SPLIT_MANIFEST_PATH" && "$TAIL_RATIO" != "0" && "$TAIL_RATIO" != "0.0" ]]; then
    echo "[parallel-eval] split manifest provided; ignoring TAIL_RATIO=$TAIL_RATIO"
    TAIL_RATIO=0
fi

mkdir -p "$OUTPUT_DIR" "$WORK_ROOT"
WORK_DIR="$(mktemp -d "$WORK_ROOT/run.XXXXXX")"

echo "[parallel-eval] mode=$EVAL_MODE"
echo "[parallel-eval] gpus=${GPUS[*]}"
echo "[parallel-eval] variants=${VARIANTS[*]}"
echo "[parallel-eval] weights=$SAVE_DIR"
echo "[parallel-eval] output=$OUTPUT_CSV"

PIDS=()
EVAL_SUCCEEDED=0
stop_workers() {
    local pid
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
cleanup() {
    stop_workers
    if [[ "$EVAL_SUCCEEDED" == "1" ]]; then
        rm -rf "$WORK_DIR"
    else
        echo "[parallel-eval] retained diagnostics: $WORK_DIR" >&2
    fi
}
handle_signal() {
    exit 130
}
trap cleanup EXIT
trap handle_signal INT TERM

run_worker() {
    local gpu="$1"
    local variants="$2"
    local output_csv="$3"
    local output_json="$4"
    local shard_index="${5:-0}"
    local num_shards="${6:-1}"
    local -a command=(
        "$PY" "$EVALUATOR"
        --variants "$variants"
        --weight_prefix "$WEIGHT_PREFIX"
        --save_dir "$SAVE_DIR"
        --data_path "$DATA_PATH"
        --tokenizer_path "$TOKENIZER_PATH"
        --hidden_size "$HIDDEN_SIZE"
        --num_hidden_layers "$NUM_HIDDEN_LAYERS"
        --num_attention_heads "$NUM_ATTENTION_HEADS"
        --num_key_value_heads "$NUM_KEY_VALUE_HEADS"
        --head_dim "$HEAD_DIM"
        --intermediate_size "$INTERMEDIATE_SIZE"
        --max_position_embeddings "$MAX_POSITION_EMBEDDINGS"
        --rope_theta "$ROPE_THETA"
        --rms_norm_eps "$RMS_NORM_EPS"
        --max_seq_len "$MAX_SEQ_LEN"
        --embedding_variant_rank "$EMBEDDING_VARIANT_RANK"
        --lm_head_bias "$LM_HEAD_BIAS"
        --use_moe "$USE_MOE"
        --tail_ratio "$TAIL_RATIO"
        --max_samples "$MAX_SAMPLES"
        --start_index "$START_INDEX"
        --max_batches "$MAX_BATCHES"
        --batch_size "$EVAL_BATCH_SIZE"
        --num_workers "$EVAL_NUM_WORKERS"
        --dtype "$EVAL_DTYPE"
        --device "$EVAL_DEVICE"
        --shard_index "$shard_index"
        --num_shards "$num_shards"
        --output_csv "$output_csv"
        --output_json "$output_json"
    )
    if [[ -n "$SPLIT_MANIFEST_PATH" ]]; then
        command+=(--split_manifest_path "$SPLIT_MANIFEST_PATH")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}"
}

wait_for_workers() {
    local failed=0
    local pid
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "[parallel-eval] worker pid=$pid failed" >&2
            failed=1
        fi
    done
    PIDS=()
    return "$failed"
}

if [[ "$EVAL_MODE" == "variants" ]]; then
    declare -a ASSIGNMENTS=()
    for ((i = 0; i < ${#VARIANTS[@]}; i++)); do
        bucket=$((i % ${#GPUS[@]}))
        if [[ -n "${ASSIGNMENTS[$bucket]:-}" ]]; then
            ASSIGNMENTS[$bucket]+=","
        fi
        ASSIGNMENTS[$bucket]+="${VARIANTS[$i]}"
    done

    for ((i = 0; i < ${#GPUS[@]}; i++)); do
        variants="${ASSIGNMENTS[$i]:-}"
        [[ -z "$variants" ]] && continue
        gpu="${GPUS[$i]}"
        prefix="$WORK_DIR/worker_$i"
        echo "[parallel-eval] gpu=$gpu variants=$variants"
        run_worker "$gpu" "$variants" "$prefix.csv" "$prefix.json" \
            > "$prefix.log" 2>&1 &
        PIDS+=("$!")
    done
    if ! wait_for_workers; then
        echo "[parallel-eval] variant workers failed; see $WORK_DIR/worker_*.log" >&2
        exit 1
    fi
else
    num_shards="${#GPUS[@]}"
    for variant in "${VARIANTS[@]}"; do
        echo "[parallel-eval] variant=$variant shards=$num_shards"
        for ((i = 0; i < num_shards; i++)); do
            gpu="${GPUS[$i]}"
            prefix="$WORK_DIR/shard_${variant}_$i"
            run_worker "$gpu" "$variant" "$prefix.csv" "$prefix.json" "$i" "$num_shards" \
                > "$prefix.log" 2>&1 &
            PIDS+=("$!")
        done
        if ! wait_for_workers; then
            echo "[parallel-eval] shards failed for $variant; see $WORK_DIR/shard_${variant}_*.log" >&2
            exit 1
        fi
    done
fi

if [[ "$EVAL_MODE" == "variants" ]]; then
    mapfile -t RESULT_FILES < <(find "$WORK_DIR" -maxdepth 1 -type f -name 'worker_*.json' | sort)
else
    mapfile -t RESULT_FILES < <(find "$WORK_DIR" -maxdepth 1 -type f -name 'shard_*.json' | sort)
fi

"$PY" - "$EVAL_MODE" "$VARIANTS_CSV" "${#GPUS[@]}" "$OUTPUT_CSV" "$OUTPUT_JSON" "$SUMMARY_DIR" "${RESULT_FILES[@]}" <<'PY'
import csv
import glob
import json
import math
import os
import sys

mode, variants_csv, gpu_count, output_csv, output_json, summary_dir = sys.argv[1:7]
paths = sys.argv[7:]
expected = [item.strip().lower() for item in variants_csv.split(",") if item.strip()]
gpu_count = int(gpu_count)

raw_rows = []
for path in paths:
    with open(path, "r", encoding="utf-8") as handle:
        raw_rows.extend(json.load(handle))

if mode == "variants":
    grouped = {}
    for row in raw_rows:
        variant = row["variant"]
        if variant in grouped:
            raise RuntimeError(f"duplicate result for {variant}")
        grouped[variant] = row
else:
    shard_groups = {}
    for row in raw_rows:
        shard_groups.setdefault(row["variant"], []).append(row)
    grouped = {}
    for variant, shards in shard_groups.items():
        if len(shards) != gpu_count:
            raise RuntimeError(
                f"expected {gpu_count} shards for {variant}, found {len(shards)}"
            )
        total_tokens = sum(int(row["tokens"]) for row in shards)
        total_nll = sum(float(row["loss"]) * int(row["tokens"]) for row in shards)
        loss = total_nll / max(total_tokens, 1)
        first = shards[0]
        grouped[variant] = {
            "variant": variant,
            "loss": loss,
            "ppl": math.exp(loss) if loss < 50 else float("inf"),
            "tokens": total_tokens,
            "sequences": sum(int(row["sequences"]) for row in shards),
            "seconds": max(float(row["seconds"]) for row in shards),
            "data_path": first.get("data_path", ""),
            "start_index": first.get("start_index", 0),
            "max_samples": first.get("max_samples", 0),
        }

missing = [variant for variant in expected if variant not in grouped]
extra = sorted(set(grouped) - set(expected))
if missing or extra:
    raise RuntimeError(f"result mismatch: missing={missing}, extra={extra}")

losses = {variant: float(grouped[variant]["loss"]) for variant in expected}
s1_loss = losses.get("s1")
s2_loss = losses.get("s2")
rows = []
for variant in expected:
    source = grouped[variant]
    loss = losses[variant]
    rows.append({
        "variant": variant,
        "loss": f"{loss:.6f}",
        "ppl": f"{math.exp(loss):.6f}" if loss < 50 else "inf",
        "delta_vs_s1": "" if s1_loss is None else f"{loss - s1_loss:+.6f}",
        "delta_vs_s2": "" if s2_loss is None else f"{loss - s2_loss:+.6f}",
        "tokens": int(source["tokens"]),
        "sequences": int(source["sequences"]),
        "seconds": f"{float(source['seconds']):.2f}",
        "data_path": source.get("data_path", ""),
        "start_index": source.get("start_index", 0),
        "max_samples": source.get("max_samples", 0),
    })

fieldnames = [
    "variant", "loss", "ppl", "delta_vs_s1", "delta_vs_s2",
    "tokens", "sequences", "seconds", "data_path", "start_index", "max_samples",
]
os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with open(output_json, "w", encoding="utf-8") as handle:
    json.dump(rows, handle, ensure_ascii=False, indent=2)

if summary_dir:
    summary_rows = []
    pattern = os.path.join(summary_dir, "seed*", "eval_pretrain_loss.json")
    for path in sorted(glob.glob(pattern)):
        seed = os.path.basename(os.path.dirname(path))
        with open(path, "r", encoding="utf-8") as handle:
            for row in json.load(handle):
                summary_rows.append({"seed": seed, **row})

    def summary_key(row):
        variant = row["variant"]
        variant_number = int(variant[1:]) if variant.startswith("s") else 10_000
        seed = row["seed"]
        seed_number = int(seed[4:]) if seed.startswith("seed") else 10_000
        return variant_number, seed_number

    summary_rows.sort(key=summary_key)
    summary_csv = os.path.join(summary_dir, "eval_pretrain_loss_all_seeds.csv")
    summary_json = os.path.join(summary_dir, "eval_pretrain_loss_all_seeds.json")
    summary_fieldnames = ["seed", *fieldnames]
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, ensure_ascii=False, indent=2)
    print(f"[parallel-eval] wrote {summary_csv}")
    print(f"[parallel-eval] wrote {summary_json}")

print(f"[parallel-eval] wrote {output_csv}")
print(f"[parallel-eval] wrote {output_json}")
for row in rows:
    print(
        f"[parallel-eval] {row['variant']}: loss={row['loss']} "
        f"ppl={row['ppl']} tokens={row['tokens']}"
    )
PY

EVAL_SUCCEEDED=1
echo "[parallel-eval] done"
