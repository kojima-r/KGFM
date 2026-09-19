#!/usr/bin/env bash
# Resume an interrupted dataset-size run. Cells whose JSON already exists are
# skipped; the rest continue from their last checkpoint.
#
#   bash benchmark_scaling_data/resume_data.sh latest
#   bash benchmark_scaling_data/resume_data.sh latest --config benchmark_scaling_data/config_data_ngram.yaml
#
# The config must match the interrupted run; the scratch sweep is the default
# because it is the one long enough to be interrupted.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TARGET="${1:-latest}"; shift || true

kgfm bench run --config benchmark_scaling_data/config_data_scratch.yaml \
    --resume "$TARGET" --results-root benchmark_scaling_data/results/chembl "$@"
kgfm report --out-dir "$TARGET" --results-root benchmark_scaling_data/results/chembl
python benchmark_scaling_data/report_data.py --out-dir "$TARGET"
