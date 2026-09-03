#!/usr/bin/env bash
# One complete pass over every row in list_chembl/train.txt
# (benchmarks/config_xxlarge.yaml): 674,265,105 examples, 85 files, 105 GiB,
# using the architecture that won the 28-cell comparison.
#
#   encoder  gte-large   head  mlp   mode  fine-tune   B=256   proj_dim=256
#
# That cell scored the best filtered MRR of the 28 (0.4641, vs 0.4331 for the
# ngram baseline) at 6000 steps; this runs the same architecture over the whole
# corpus. Runtime is not the constraint here:
#
# The length is declared as `max_epoch: 1.0`, not as a step count, so it stays
# one epoch whatever the batch size and GPU count work out to:
#   1 GPU  B=256  ->  2,633,849 steps   ~249 h = 10.4 days
#   2 GPUs B=256  ->  1,316,925 steps   ~125 h =  5.2 days
# Two GPUs halve the wall clock rather than doubling the data, because train
# files are sharded files[rank::world_size] and each rank reads a disjoint half:
#   bash benchmarks/run_chembl_xxlarge.sh --nproc 2
#
# A same-day full pass instead (5th place, MRR 0.4331, ~11 h). No step count to
# recompute — max_epoch absorbs the larger batch:
#   bash benchmarks/run_chembl_xxlarge.sh \
#       --encoders ngram --heads linear --batch-size 512
#
# To get the ngram baseline into the same report, run it as a SECOND pass into
# the same run directory (+11 h, ~4% of the run). It cannot go in the first
# pass: `heads` is a global axis, so --encoders gte-large,ngram would give ngram
# an mlp head — and ngram+mlp was the worst of all 28 cells (0.2539) while
# ngram+linear was 5th (0.4331). --resume skips the cells already done and runs
# only the new one; `kgfm report` globs the directory, so the table combines
# them:
#   bash benchmarks/run_chembl_xxlarge.sh --resume latest \
#       --encoders ngram --heads linear --batch-size 512
#
# Resume is cheap relative to the length — checkpoints land every 20k steps, so
# at most ~1.9 h is lost:
#   bash benchmarks/resume_chembl.sh latest
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_xxlarge.yaml "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
