#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PY="${PY:-/home/wz/anaconda3/envs/torch24/bin/python}"
SEEDS="${SEEDS:-123 2026}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/results/minimind-small-fixedinit-randsplit}"

for seed in $SEEDS; do
    run_id="seed${seed}_full_$(date +%Y%m%d_%H%M%S)"
    echo "================================================================"
    echo "[dynamic-centering-all] seed=$seed run_id=$run_id"
    echo "================================================================"
    SEED="$seed" RUN_ID="$run_id" RESULT_ROOT="$RESULT_ROOT" bash "$SCRIPT_DIR/run_seed.sh"
done

if [[ -f "$RESULT_ROOT/seed42/comparison.csv" && \
      -f "$RESULT_ROOT/seed123/comparison.csv" && \
      -f "$RESULT_ROOT/seed2026/comparison.csv" ]]; then
    "$PY" "$SCRIPT_DIR/summarize_three_seeds.py" \
        --seed_result "42=$RESULT_ROOT/seed42/comparison.csv" \
        --seed_result "123=$RESULT_ROOT/seed123/comparison.csv" \
        --seed_result "2026=$RESULT_ROOT/seed2026/comparison.csv" \
        --output_dir "$RESULT_ROOT"
fi
