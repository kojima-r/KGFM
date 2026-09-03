#!/usr/bin/env bash
# Scaling study over FROZEN encoders, including the 7B one that cannot be
# fine-tuned at all. See benchmark_scaling/config_scaling_frozen.yaml.
#
# Only the projection head trains, so compute is 2*N*T (forward only) for the
# encoder plus 6*N_head*V for the head — `kgfm scaling` applies that split
# automatically from the cell's freeze flag. Treating a frozen encoder as 6NT
# would overstate its compute threefold.
#
# e5-mistral-7b is a ~14 GB download on first use:
#   python -c "from kgfm.encoders import make_encoder; make_encoder('e5-mistral-7b', freeze_encoder=True)"
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling/config_scaling_frozen.yaml "$@"
kgfm report  --out-dir latest --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/chembl
