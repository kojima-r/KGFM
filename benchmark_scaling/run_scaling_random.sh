#!/usr/bin/env bash
# Model-size scaling law on the RANDOM multi-source corpus (list_random/).
#
# Same axis and same trainer as run_scaling_scratch.sh; the corpus is what
# changes. Compare the exponent against the ChEMBL run restricted to the same
# five sizes and truncated to the same 24k steps — see the header of
# config_scaling_random.yaml for that baseline, and
# benchmark_scaling/compare_corpora.py to recompute it.
#
# Seven sizes, 4.4M -> 335.4M, ~16 h of GPU time (~8 h split over two cards).
# `--resume <run>` skips cells whose JSON already exists, so the two largest
# can be added to a finished smaller run rather than repeating it:
#   bash benchmark_scaling/run_scaling_random.sh --resume <timestamp>
#
# Per-size learning rates come from the three probe scripts
# (run_lr_probe_random.sh, _extend.sh, _large.sh); run those first if
# config_scaling_random.yaml's `cells:` block is incomplete, because the
# fallback (train.SCRATCH_LR = 1e-4) collapses scratch-base outright on this
# corpus.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmark_scaling/config_scaling_random.yaml "$@"
kgfm report  --out-dir latest --results-root benchmark_scaling/results/random
kgfm scaling --out-dir latest --results-root benchmark_scaling/results/random
