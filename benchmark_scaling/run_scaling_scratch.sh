#!/usr/bin/env bash
# Scaling study over randomly-initialized encoders — size varies, pretraining
# does not. Compare its exponent against run_scaling.sh's: the difference is
# what pretraining contributes.
#
# Per-size learning rates come from benchmark_scaling/lr_probe.py; run that
# first if config_scaling_scratch.yaml's `cells:` block is still empty.
set -euo pipefail
cd "$(dirname "$0")/.."

kgfm bench run --config benchmark_scaling/config_scaling_scratch.yaml "$@"
kgfm report  --out-dir latest --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/chembl
