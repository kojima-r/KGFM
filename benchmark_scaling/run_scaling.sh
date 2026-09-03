#!/usr/bin/env bash
# Scaling study: validation loss vs compute, across model sizes.
# See benchmark_scaling/config_scaling.yaml for the design and the cost.
#
# Two commands, same split as benchmarks/: `kgfm bench run` does the training
# (it is the same pipeline — a scaling run is just a sweep whose axis is model
# size), and `kgfm scaling` turns the resulting logs into the compute-vs-loss
# plots. Nothing here re-trains anything to produce a plot.
#
#   results, logs, report -> benchmark_scaling/results/chembl/<timestamp>_scaling/
#
# Five model sizes, ~1.1 h on one H200. Extend every line to the right with:
#   bash benchmark_scaling/run_scaling.sh --max-steps 20000
#
# Two GPUs (halves wall clock; the negative count stays per-rank):
#   bash benchmark_scaling/run_scaling.sh --nproc 2
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling/config_scaling.yaml "$@"
kgfm report  --out-dir latest --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/chembl
