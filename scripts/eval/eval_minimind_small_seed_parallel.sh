#!/usr/bin/env bash
# Parallel eval for the current MiniMind small fixedinit/randsplit run.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

PY="${PY:-/home/wz/anaconda3/envs/torch24/bin/python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_PATH="${CUDA_PATH:-/home/wz/anaconda3/envs/torch24}"
export CUDA_HOME="${CUDA_HOME:-/home/wz/anaconda3/envs/torch24}"

RUN_ID="${RUN_ID:-20260602_142805}"
SEED="${SEED:-42}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
VARIANTS_CSV="${VARIANTS:-s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13}"
WEIGHT_PREFIX_BASE="${WEIGHT_PREFIX_BASE:-pretrain_v2_fixedinit_randsplit}"
if [[ -z "${WEIGHT_PREFIX+x}" ]]; then
  WEIGHT_PREFIX="${WEIGHT_PREFIX_BASE}_seed${SEED}"
fi
SAVE_DIR="${SAVE_DIR:-weights/final}"
DATA_PATH="${DATA_PATH:-dataset/minimind/pretrain_t2t.jsonl}"
TOKENIZER_PATH="${TOKENIZER_PATH:-model}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-data_splits/minimind_pretrain_t2t_random_eval_0.01_seed20260602.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"

SEED_DIR="${SEED_DIR:-logs/minimind-small-fixedinit/${RUN_ID}/seed${SEED}}"
WORK_DIR="${SEED_DIR}/eval_parallel"
mkdir -p "$WORK_DIR"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a VARIANTS <<< "$VARIANTS_CSV"
GPU_COUNT="${#GPUS[@]}"
if (( GPU_COUNT == 0 )); then
  echo "GPU_LIST is empty" >&2
  exit 1
fi

declare -a ASSIGNMENTS
for ((i = 0; i < GPU_COUNT; i++)); do
  ASSIGNMENTS[$i]=""
done

for ((i = 0; i < ${#VARIANTS[@]}; i++)); do
  bucket=$((i % GPU_COUNT))
  if [[ -n "${ASSIGNMENTS[$bucket]}" ]]; then
    ASSIGNMENTS[$bucket]+=","
  fi
  ASSIGNMENTS[$bucket]+="${VARIANTS[$i]}"
done

echo "[parallel-eval] run_id=$RUN_ID seed=$SEED"
echo "[parallel-eval] weight_prefix=$WEIGHT_PREFIX"
echo "[parallel-eval] split_manifest=$SPLIT_MANIFEST_PATH"
echo "[parallel-eval] work_dir=$WORK_DIR"

PIDS=()
for ((i = 0; i < GPU_COUNT; i++)); do
  gpu="${GPUS[$i]}"
  variants="${ASSIGNMENTS[$i]}"
  if [[ -z "$variants" ]]; then
    continue
  fi

  log_file="${WORK_DIR}/gpu${gpu}.log"
  out_csv="${WORK_DIR}/gpu${gpu}.csv"
  out_json="${WORK_DIR}/gpu${gpu}.json"

  echo "[parallel-eval] gpu=$gpu variants=$variants log=$log_file"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" results/eval_pretrain_loss.py \
      --variants "$variants" \
      --weight_prefix "$WEIGHT_PREFIX" \
      --save_dir "$SAVE_DIR" \
      --data_path "$DATA_PATH" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --tail_ratio 0.0 \
      --split_manifest_path "$SPLIT_MANIFEST_PATH" \
      --max_samples 0 \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --lm_head_bias 1 \
      --device cuda:0 \
      --output_csv "$out_csv" \
      --output_json "$out_json"
  ) > "$log_file" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "[parallel-eval] one or more workers failed; see $WORK_DIR/gpu*.log" >&2
  exit 1
fi

"$PY" - "$WORK_DIR" "${SEED_DIR}/eval_pretrain_loss.csv" "${SEED_DIR}/eval_pretrain_loss.json" <<'PY'
import csv
import glob
import json
import math
import os
import sys

work_dir, output_csv, output_json = sys.argv[1:4]
rows = []
for path in sorted(glob.glob(os.path.join(work_dir, "gpu*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        rows.extend(json.load(f))

def variant_key(row):
    variant = row["variant"]
    try:
        return int(variant[1:])
    except Exception:
        return 10_000

rows.sort(key=variant_key)
loss_by_variant = {row["variant"]: float(row["loss"]) for row in rows}
s1_loss = loss_by_variant.get("s1")
s2_loss = loss_by_variant.get("s2")
for row in rows:
    loss = float(row["loss"])
    row["ppl"] = math.exp(loss) if loss < 50 else float("inf")
    row["delta_vs_s1"] = "" if s1_loss is None else loss - s1_loss
    row["delta_vs_s2"] = "" if s2_loss is None else loss - s2_loss

os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
fieldnames = [
    "variant", "loss", "ppl", "delta_vs_s1", "delta_vs_s2",
    "tokens", "sequences", "seconds", "data_path", "start_index", "max_samples",
]
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"[parallel-eval] wrote {output_csv}")
print(f"[parallel-eval] wrote {output_json}")
PY

echo "[parallel-eval] done"
