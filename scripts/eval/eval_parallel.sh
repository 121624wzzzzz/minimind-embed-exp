#!/usr/bin/env bash
# Run eval in parallel across 8 GPUs.
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$ROOT"

PY="/home/wz/anaconda3/envs/torch24/bin/python"
export PYTHONPATH="$ROOT"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_PATH=/home/wz/anaconda3/envs/torch24
export CUDA_HOME=/home/wz/anaconda3/envs/torch24

SEEDS=(42 123 2026)
VARIANTS=(s1 s2 s3 s4 s6 s12)

TASKS=()
for seed in "${SEEDS[@]}"; do
  for v in "${VARIANTS[@]}"; do
    TASKS+=("${seed}:${v}")
  done
done

echo "Total tasks: ${#TASKS[@]}"
RUNNING=0
for task in "${TASKS[@]}"; do
  seed="${task%%:*}"
  v="${task##*:}"
  gpu=$((RUNNING % 8))

  echo "[gpu:$gpu] seed=$seed $v"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" results/eval_pretrain_loss.py \
    --data_path dataset/minimind/pretrain_t2t.jsonl \
    --tokenizer_path model \
    --save_dir "weights/final/minimind-large/seed${seed}" \
    --weight_prefix "" \
    --variants "$v" \
    --hidden_size 1024 \
    --num_hidden_layers 16 \
    --max_samples 0 \
    --tail_ratio 0.01 \
    --batch_size 32 \
    > "logs/minimind/eval_seed${seed}_${v}.log" 2>&1 &

  RUNNING=$((RUNNING + 1))
  # Wait every 8 tasks to keep concurrency at 8
  if (( RUNNING % 8 == 0 )); then
    wait
  fi
done

wait
echo "=== ALL EVAL DONE ==="
echo "Files: $(ls logs/minimind/eval_seed*.log 2>/dev/null | wc -l)"
