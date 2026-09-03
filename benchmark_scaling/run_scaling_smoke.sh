#!/usr/bin/env bash
# Smoke test: proves training -> logs -> report -> scaling plots end to end in
# a few minutes, with three small cells. Run this before committing to the
# real study.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling/config_scaling_small.yaml --skip viz "$@"
kgfm report  --out-dir latest --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/chembl
