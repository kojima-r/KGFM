#!/usr/bin/env bash
# Per-size learning-rate probe on the RANDOM multi-source list (list_random/).
#
# WHY THIS EXISTS SEPARATELY FROM run_lr_probe.sh
# That script probes on ChEMBL. The rates it found are properties of
# (architecture, corpus), not of the architecture alone — so reusing them on a
# different corpus would confound "this corpus scales differently" with "the
# ChEMBL rate happened to suit this corpus worse". That is exactly the failure
# benchmark_scaling/README.md documents at length for the size axis, and it
# applies just as much when the corpus changes instead of the model.
#
# Grid brackets every optimum the ChEMBL probe found for these five sizes
# (1e-5 .. 1e-4), with 3e-6 below so the smallest tolerated rate is interior
# rather than an edge.
#
# Writes benchmark_scaling/results/lr_probe_random/summary.txt, whose `cells:`
# block goes into config_scaling_random.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=benchmark_scaling/results/lr_probe_random
LRS=${LRS:-3e-6,1e-5,3e-5,1e-4}
STEPS=${STEPS:-2000}
LISTS="--train-list list_random/train.txt --valid-list list_random/valid.txt --test-list list_random/test.txt"

python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-tiny,scratch-mini,scratch-small,scratch-medium \
    --lrs "$LRS" --gpus 0,0,1,1 --steps "$STEPS" --eval-every 500 \
    --threads 6 --out-dir "$OUT" $LISTS "$@"

# bert-base-sized cells: two at a time, not four — encode_triple pushes 3B
# sequences per step and a 110M encoder at B=256 needs a large fraction of one
# H200 by itself.
python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-base \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "${OUT}_base" $LISTS "$@"

python -u benchmark_scaling/summarize_lr_probe.py "$OUT" "${OUT}_base"
