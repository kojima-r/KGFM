#!/usr/bin/env bash
# Full ChEMBL comparison at the middle scale (benchmarks/config_middle.yaml).
# Roughly 4-6 h on a single H200; see the config for the breakdown.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_middle.yaml "$@"
kgfm-ultra  --out-dir latest
kgfm-motif  --out-dir latest
kgfm report --out-dir latest
