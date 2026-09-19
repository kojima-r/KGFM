#!/usr/bin/env bash
# Upward extension of the LR grid on the random multi-source corpus.
#
# WHY THIS EXISTS, AND WHY IT GOES *UP* WHILE run_lr_probe_extend.sh GOES DOWN
# On ChEMBL the first grid (1e-4 .. 3e-3) returned its bottom value for every
# small size, so that extension moved down. On list_random the first grid
# (3e-6 .. 1e-4) returns its *top* value: measured at 2000 steps,
#   scratch-tiny  3e-6 4.7489 | 1e-5 4.2007 | 3e-5 3.5048 | 1e-4 [3.2943]
#   scratch-mini  3e-6 4.4018 | 1e-5 3.7590 | 3e-5 3.3329 | 1e-4 [3.1472]
# A winner on the edge of the grid is a bound, not an optimum, so the grid has
# to move until the winner is interior.
#
# That the direction is opposite to ChEMBL's is itself the finding this probe
# exists to catch: the tolerated rate is a property of (architecture, corpus),
# and this corpus tolerates roughly a decade more than ChEMBL does. Reusing
# ChEMBL's rates here would have run every cell an order of magnitude too slow.
#
# Results merge with the first grid: summarize_lr_probe.py reads every
# directory given and builds one table.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=benchmark_scaling/results/lr_probe_random
OUT=${BASE}_high
LRS=${LRS:-3e-4,1e-3}
STEPS=${STEPS:-2000}
ENC=${ENC:-scratch-tiny,scratch-mini,scratch-small,scratch-medium}
LISTS="--train-list list_random/train.txt --valid-list list_random/valid.txt --test-list list_random/test.txt"

python -u benchmark_scaling/lr_probe.py \
    --encoders "$ENC" \
    --lrs "$LRS" --gpus 0,0,1,1 --steps "$STEPS" --eval-every 500 \
    --threads 6 --out-dir "$OUT" $LISTS "$@"

python -u benchmark_scaling/lr_probe.py \
    --encoders scratch-base \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "${OUT}_base" $LISTS "$@"

python -u benchmark_scaling/summarize_lr_probe.py \
    "$BASE" "${BASE}_base" "$OUT" "${OUT}_base"
