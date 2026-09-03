#!/usr/bin/env bash
# Resume an interrupted scaling run. Cells whose JSON already exists are
# skipped; the rest continue from their last checkpoint.
#   bash benchmark_scaling/resume_scaling.sh latest
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TARGET="${1:-latest}"; shift || true

kgfm bench run --config benchmark_scaling/config_scaling.yaml \
    --resume "$TARGET" --results-root benchmark_scaling/results/chembl "$@"
kgfm report  --out-dir "$TARGET" --results-root benchmark_scaling/results/chembl
kgfm scaling --out-dir "$TARGET" --results-root benchmark_scaling/results/chembl
