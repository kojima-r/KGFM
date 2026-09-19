#!/usr/bin/env bash
# Scorer x head-mode comparison: 8 cells (2 modes x 4 scorers), ~3 h.
#
# Three commands, same split as the other studies: `kgfm bench run` trains,
# `kgfm report` writes the ordinary flat table, and report_scorer.py turns the
# same results into the 2-D grid the comparison actually needs.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
kgfm bench run --config benchmark_scorer/config_scorer.yaml "$@"
kgfm report --out-dir latest --results-root benchmark_scorer/results/chembl
python benchmark_scorer/report_scorer.py --out-dir latest
