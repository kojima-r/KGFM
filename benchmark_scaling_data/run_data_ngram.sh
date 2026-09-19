#!/usr/bin/env bash
# Dataset-size scaling law with the ngram encoder — the cheap sweep (~2 h).
#
# Same three commands as benchmark_scaling/: `kgfm bench run` trains,
# `kgfm report` writes the ordinary benchmark report, and report_data.py turns
# the same cell logs into (unique data, loss) coordinates. Nothing re-trains
# anything to produce a plot.
#
#   results -> benchmark_scaling_data/results/chembl/<timestamp>_data_scaling_ngram/
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling_data/config_data_ngram.yaml "$@"
kgfm report --out-dir latest --results-root benchmark_scaling_data/results/chembl
python benchmark_scaling_data/report_data.py --out-dir latest
