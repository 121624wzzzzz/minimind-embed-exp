#!/usr/bin/env bash
set -euo pipefail

ROOT="$(readlink -f "$(dirname "$(readlink -f "$0")")/../..")"
MANIFEST="$ROOT/data_splits/fineedu_gpt2_6b_random_eval_0.01_seed20260602.json"

if [[ ! -f "$MANIFEST" ]]; then
    echo "[randsplit] missing manifest: $MANIFEST" >&2
    exit 1
fi

cd "$ROOT"

for SEED_VALUE in 42 123 2026; do
    echo "===== FineEdu GPT2 random split seed ${SEED_VALUE} start $(date '+%F %T') ====="
    SEED="$SEED_VALUE" \
    VARIANTS="s1,s2,s3,s4,s5,s6,s7,s11,s12,s13" \
    GPUS="0,1,2,3,4,5,6,7" \
    DATA_PATH="../dataset/fineweb_edu/packed/gpt2_6b_seq340" \
    TOKENIZER_PATH="gpt2" \
    SAVE_DIR="$ROOT/weights/final/fineedu-gpt2-randsplit-6b/seed${SEED_VALUE}" \
    CHECKPOINT_DIR="$ROOT/weights/resume/fineedu-gpt2-randsplit-6b/seed${SEED_VALUE}" \
    WEIGHT_PREFIX="" \
    TRAIN_SPLIT_RATIO="1.0" \
    SPLIT_MANIFEST_PATH="$MANIFEST" \
    BATCH_SIZE="80" \
    ACCUMULATION_STEPS="1" \
    MAX_SEQ_LEN="340" \
    EPOCHS="1" \
    LEARNING_RATE="5e-4" \
    LM_HEAD_BIAS="1" \
    RANK="32" \
    LOG_INTERVAL="100" \
    SAVE_INTERVAL="5000" \
    NUM_WORKERS="8" \
    GRAD_LOG_INTERVAL="1000" \
    GRAD_SAVE_TENSORS="0" \
    SKIP_COMPLETED="1" \
    RUN_EVAL="1" \
    EVAL_BATCH_SIZE="32" \
    LOG_DIR="$ROOT/logs/fineedu-gpt2-6b-randsplit-seed${SEED_VALUE}" \
    bash scripts/train/train_fineedu_gpt2_pretrain.sh
    echo "===== FineEdu GPT2 random split seed ${SEED_VALUE} done $(date '+%F %T') ====="
done
