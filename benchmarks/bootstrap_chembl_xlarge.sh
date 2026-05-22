#!/usr/bin/env bash
# bootstrap_chembl_large.sh — thin wrapper around bootstrap_chembl.sh that
# raises the dataset / training / filter / batch caps to a more realistic
# scale so the resulting Hits@K / MRR numbers are less noisy.
#
# Why a wrapper instead of a fork: bootstrap_chembl.sh's argument parser is
# last-write-wins, so any flag the user passes here is appended after our
# enlarged defaults and overrides them. That keeps this script in lockstep
# with future changes to the original — only the defaults differ.
#
# See benchmarks/README.md ("全コーパスに対する --max-* のカバー率" and
# "より大きいスケールで走らせる" sections) for the exact default values
# and the corpus-coverage estimates these defaults imply.
#
# Usage:
#   bash benchmarks/bootstrap_chembl_large.sh [extra bootstrap_chembl flags]
#
# Examples:
#   bash benchmarks/bootstrap_chembl_large.sh
#   bash benchmarks/bootstrap_chembl_large.sh --kgfm-encoders ngram
#   bash benchmarks/bootstrap_chembl_large.sh --batch-size 2048 --max-steps 5000
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$HERE/bootstrap_chembl.sh" \
    --max-train         5000000 \
    --max-valid          100000 \
    --max-test           100000 \
    --max-steps           20000 \
    --batch-size          4048 \
    --max-filter-tails  200000 \
    --max-filter-rows  5000000 \
    "$@"
