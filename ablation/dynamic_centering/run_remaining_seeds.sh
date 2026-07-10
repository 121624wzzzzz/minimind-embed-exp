#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SEEDS="${SEEDS:-123 2026}"

for seed in $SEEDS; do
    run_id="seed${seed}_full_$(date +%Y%m%d_%H%M%S)"
    echo "================================================================"
    echo "[dynamic-centering-all] seed=$seed run_id=$run_id"
    echo "================================================================"
    SEED="$seed" RUN_ID="$run_id" bash "$SCRIPT_DIR/run_seed42.sh"
done
