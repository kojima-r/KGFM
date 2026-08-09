#!/usr/bin/env bash
# Full ChEMBL comparison at the largest scale (benchmarks/config_xlarge.yaml).
#
#   bash benchmarks/run_chembl_xlarge.sh --nproc 2 --per-device-train-batch-size 1024
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_xlarge.yaml "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
