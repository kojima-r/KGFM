#!/usr/bin/env bash
# Per-size learning-rate probe for the PRETRAINED scaling family.
#
# Same purpose as run_lr_probe.sh, different rate grid: these models start from
# useful weights, so the range that matters is an order of magnitude lower.
# 3e-5 is `train.TRANSFORMER_LR`, the single default every cell currently gets;
# this measures whether it actually suits a 4.4M model as well as a 110M one.
#
# 1e-3 is deliberately absent: it is the rate measured to collapse BERT
# outright (loss 5.5498 -> 5.5466 over 200 steps, cosine similarity 1.000000
# between unrelated texts), so spending four cells on it buys nothing.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmark_scaling/results/lr_probe_pretrained
LRS=${LRS:-1e-5,3e-5,1e-4,3e-4}
STEPS=${STEPS:-2000}

python -u benchmark_scaling/lr_probe.py \
    --encoders bert-tiny,bert-mini,bert-small,bert-medium \
    --lrs "$LRS" --gpus 0,0,1,1 --steps "$STEPS" --eval-every 500 \
    --threads 6 --out-dir "$OUT" "$@"

python -u benchmark_scaling/lr_probe.py \
    --encoders mpnet \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "${OUT}_mpnet" "$@"

python -u benchmark_scaling/summarize_lr_probe.py "$OUT" "${OUT}_mpnet"
