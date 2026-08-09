#!/usr/bin/env bash
# Full ChEMBL comparison at a scale where the numbers are less noisy.
# See benchmarks/config_large.yaml for the settings.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_large.yaml "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
