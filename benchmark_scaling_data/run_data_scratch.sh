#!/usr/bin/env bash
# Dataset-size scaling law with a randomly-initialised 41M transformer.
# The headline sweep: ~11 h on one H200 (8 cells x ~1.3 h).
#
# Compare its exponent with run_data_ngram.sh's: same axis, and the difference
# is what a real encoder does with a finite pool that a lookup table does not.
# Compare its FLOOR with benchmark_scaling/run_scaling_scratch.sh at the same
# encoder — that run is this run's uncapped cell with a bigger step budget.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling_data/config_data_scratch.yaml "$@"
kgfm report --out-dir latest --results-root benchmark_scaling_data/results/chembl
python benchmark_scaling_data/report_data.py --out-dir latest
