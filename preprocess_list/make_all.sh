#!/usr/bin/env bash
# Regenerate — or just verify — the list_*/ directories checked into the repo.
#
#   bash preprocess_list/make_all.sh            # verify only (default, writes nothing)
#   bash preprocess_list/make_all.sh --write    # actually rewrite the lists
#
# Verify-only is the default because these lists define the train/test boundary
# of every result in benchmarks/results/. Rewriting one silently re-splits the
# corpus, so it takes an explicit flag.
#
# list_small/ is deliberately absent from both modes: it is a hand-picked
# 5-file smoke set (3 large amrportal files for train, 2 tiny biomodels files
# for valid/test) that no single rule reproduces. See preprocess_list/README.md
# for the nearest rule-based equivalent.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE=check
if [[ "${1:-}" == "--write" ]]; then MODE=write; shift; fi

run() {           # run <out-dir> <args...>
    local dir=$1; shift
    if [[ $MODE == check ]]; then
        python preprocess_list/make_lists.py "$@" --check "$dir"
    else
        python preprocess_list/make_lists.py "$@" --out-dir "$dir" --force
    fi
}

echo "### list_chembl — data/chembl, hash split (85/6/4 of 95 files)"
# The hash split is the same rule kgfm.data.split_files_three_way applies when
# no lists are given, so this list and the trainer's own fallback agree.
run list_chembl --source chembl

echo
echo "### list_large — first 60 files of amrportal+bacdive+biomodels, 40/10/10"
# Sequential, not hash: this list exists to be a fixed medium-sized slice of
# three specific sources, and the order is the split.
run list_large --source amrportal,bacdive,biomodels \
    --split sequential --max-files 60 \
    --train-files 40 --valid-files 10 --test-files 10

echo
if [[ $MODE == check ]]; then
    echo "Verified. Pass --write to regenerate."
else
    echo "Rewritten. git diff the list_*/ dirs before committing — a changed"
    echo "split means existing results are no longer comparable."
fi
