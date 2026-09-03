#!/usr/bin/env bash
# Per-size learning-rate probe for the random-init scaling family.
#
# Two phases, because concurrency is limited by GPU memory and only at the
# large end. The cells are CPU/IO-bound — the GPUs sit near 4% utilization —
# so running several at once is nearly free in wall clock, but `encode_triple`
# pushes 3B sequences per step and a 110M encoder at B=256 needs a large
# fraction of one H200 by itself. So: four at a time up to 41M, two at a time
# for bert-base-sized cells.
#
# Writes benchmark_scaling/results/lr_probe_scratch/summary.txt, whose
# `cells:` block goes straight into config_scaling_scratch.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmark_scaling/results/lr_probe_scratch
LRS=${LRS:-1e-4,3e-4,1e-3,3e-3}
STEPS=${STEPS:-2000}

python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-tiny,scratch-mini,scratch-small,scratch-medium \
    --lrs "$LRS" --gpus 0,0,1,1 --steps "$STEPS" --eval-every 500 \
    --threads 6 --out-dir "$OUT" "$@"

# Same --out-dir: lr_probe.json is rewritten from the results this invocation
# collected, so the second phase is kept in its own directory and merged by
# summarize_lr_probe.py rather than silently overwriting the first.
python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-base \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "${OUT}_base" "$@"

python -u benchmark_scaling/summarize_lr_probe.py "$OUT" "${OUT}_base"
