#!/usr/bin/env bash
# LR probe for the two LARGEST sizes on the random multi-source corpus.
#
# Separate from run_lr_probe_random.sh because these cells need a whole GPU
# each: `encode_triple` pushes 3B sequences per step and scratch-large's
# activations peak at ~102 GiB of 139.8 at B=256, so `--gpus 0,1` runs exactly
# one cell per card. The smaller sizes run four at a time in the other script.
#
# Grid: 3e-6 .. 1e-4. `scratch-base` collapses at 1e-4 on this corpus and
# tolerance falls with size, so 1e-4 is included to prove the ceiling rather
# than in the expectation of winning there — an optimum has to be interior.
# On ChEMBL these two sizes wanted 1e-5 and 3e-6; this corpus has run about a
# decade higher at every other size, so expect 3e-5 / 1e-5 here.
set -euo pipefail
cd "$(dirname "$0")/.."

# OUT and ENC are overridable so a follow-up grid lands in its own directory
# and gets merged rather than overwriting this one, e.g. the downward
# extension scratch-large needs:
#   ENC=scratch-large LRS=1e-6,3e-7 \
#     OUT=benchmark_scaling/results/lr_probe_random_large_low \
#     bash benchmark_scaling/run_lr_probe_random_large.sh
OUT=${OUT:-benchmark_scaling/results/lr_probe_random_large}
ENC=${ENC:-scratch-xl,scratch-large}
LRS=${LRS:-3e-6,1e-5,3e-5,1e-4}
STEPS=${STEPS:-2000}
LISTS="--train-list list_random/train.txt --valid-list list_random/valid.txt --test-list list_random/test.txt"

python -u benchmark_scaling/lr_probe.py \
    --encoders "$ENC" \
    --lrs "$LRS" --gpus 0,1 --steps "$STEPS" --eval-every 500 \
    --threads 12 --out-dir "$OUT" $LISTS "$@"

# Merge every random-corpus probe directory that exists, so the table is
# complete however many follow-up grids have been run.
python -u benchmark_scaling/summarize_lr_probe.py \
    $(ls -d benchmark_scaling/results/lr_probe_random \
            benchmark_scaling/results/lr_probe_random_* 2>/dev/null)
