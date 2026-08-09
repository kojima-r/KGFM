#!/usr/bin/env bash
# Full ChEMBL comparison at the smoke scale (benchmarks/config_small.yaml).
#
#   bash benchmarks/run_chembl.sh
#   bash benchmarks/run_chembl.sh --max-steps 1000 --encoders ngram
#
# Extra flags go to `kgfm bench run` (the kgfm side). To change how a baseline
# runs, call it directly instead: `kgfm-motif --out-dir latest --gpus null`.
# `kgfm bench run` repoints the `latest` symlink at the run it creates, which
# is how the later commands find it.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_small.yaml "$@"
kgfm-ultra  --out-dir latest
kgfm-motif  --out-dir latest
kgfm report --out-dir latest
