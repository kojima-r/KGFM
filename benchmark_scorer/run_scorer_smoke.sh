#!/usr/bin/env bash
# Smoke test: 8 cells (2 head modes x 4 scorers) in a few minutes.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
kgfm bench run --config benchmark_scorer/config_scorer_small.yaml --skip viz "$@"
kgfm report --out-dir latest --results-root benchmark_scorer/results/chembl
python benchmark_scorer/report_scorer.py --out-dir latest
