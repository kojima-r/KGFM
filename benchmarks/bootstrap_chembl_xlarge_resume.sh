#!/usr/bin/env bash
# bootstrap_chembl_xlarge_resume.sh — resume a previously interrupted
# bootstrap_chembl_xlarge.sh run from wherever it stopped.
#
# Steps whose output file is already present in the target run directory are
# skipped automatically:
#   prepare_chembl_kg   skipped if chembl_kg_stats.json exists
#   kgfm cell           skipped if its kgfm_*.json exists; if only the
#                       checkpoint (best.pt/final.pt) is present, training
#                       is skipped and just the eval is re-run
#   run_ultra           skipped if ultra.json exists
#   run_motif           skipped if motif.json exists
#   aggregate           always re-runs (cheap; rebuilds table.md)
#
# The xlarge defaults (max-train, batch-size, ...) are kept identical to
# bootstrap_chembl_xlarge.sh — this script just adds --resume on top.
#
# Usage:
#   bash benchmarks/bootstrap_chembl_xlarge_resume.sh                  # resume "latest"
#   bash benchmarks/bootstrap_chembl_xlarge_resume.sh 20260507T155834Z # resume a specific run
#   bash benchmarks/bootstrap_chembl_xlarge_resume.sh /abs/path/to/run # resume an explicit dir
#
# Any extra flags after the resume target are forwarded to bootstrap_chembl.sh,
# e.g.:
#   bash benchmarks/bootstrap_chembl_xlarge_resume.sh latest --skip-ultra
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# First positional argument (if not a flag) is the resume target.
TARGET="latest"
if [[ $# -gt 0 && "$1" != --* ]]; then
    TARGET=$1
    shift
fi

exec bash "$HERE/bootstrap_chembl_xlarge.sh" --resume "$TARGET" "$@"
