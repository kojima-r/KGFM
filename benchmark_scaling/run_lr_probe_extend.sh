#!/usr/bin/env bash
# Downward extension of the random-init LR grid.
#
# WHY THIS EXISTS
# The first grid (1e-4 … 3e-3) returned its *bottom* value for every small
# size: scratch-tiny 3.189 @ 1e-4 vs 3.229 @ 3e-4, scratch-mini 3.217 @ 1e-4 vs
# 3.326 @ 3e-4, with 1e-3 and above collapsing to the ln(256)=5.545
# random-guess line. A winner sitting on the edge of the grid is not an
# optimum, it is a bound — so the grid moves down until the winner is interior.
#
# Note this is the opposite of the direction a short probe usually biases
# toward. The README's caveat (short probes favour large rates, because they
# end before the large-rate curve flattens) is about fine-tuning; from random
# init the large rates do not flatten, they diverge.
#
# Results merge with the first grid: summarize_lr_probe.py reads every
# directory given and builds one table.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=benchmark_scaling/results/lr_probe_scratch
OUT=${BASE}_low
LRS=${LRS:-1e-5,3e-5}
STEPS=${STEPS:-2000}

python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-tiny,scratch-mini,scratch-small,scratch-medium \
    --lrs "$LRS" --gpus 0,0,1,1 --steps "$STEPS" --eval-every 500 \
    --threads 6 --out-dir "$OUT" "$@"

python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-base \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "${OUT}_base" "$@"

python -u benchmark_scaling/summarize_lr_probe.py \
    "$BASE" "${BASE}_base" "$OUT" "${OUT}_base"
