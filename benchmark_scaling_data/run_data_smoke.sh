#!/usr/bin/env bash
# Smoke test: training -> logs -> report -> data-scaling plots, in a few
# minutes with three tiny cells. Run this before committing to the real study.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling_data/config_data_small.yaml --skip viz "$@"
kgfm report --out-dir latest --results-root benchmark_scaling_data/results/chembl
python benchmark_scaling_data/report_data.py --out-dir latest
