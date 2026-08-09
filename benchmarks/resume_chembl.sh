#!/usr/bin/env bash
# Resume an interrupted run. Every command skips whatever it already produced.
#
#   bash benchmarks/resume_chembl.sh                    # resume 'latest'
#   bash benchmarks/resume_chembl.sh 20260507T155834Z   # a specific run
#   bash benchmarks/resume_chembl.sh latest --encoders ngram
#
# The config must match the interrupted run; xlarge is the usual case for runs
# long enough to be interrupted. Pass --config to override.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET=latest
if [[ $# -gt 0 && "$1" != -* ]]; then
    TARGET=$1
    shift
fi

kgfm bench run --config benchmarks/config_xlarge.yaml --resume "$TARGET" "$@"
kgfm-ultra  --out-dir "$TARGET" --resume
kgfm-motif  --out-dir "$TARGET" --resume
kgfm report --out-dir "$TARGET"
