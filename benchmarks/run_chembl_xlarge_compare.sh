#!/usr/bin/env bash
# Architecture comparison at xlarge data scale: which pretrained text encoder
# and which projection head actually help. See benchmarks/config_xlarge_compare.yaml.
#
# 28 cells = encoders x heads x freezes, covering every encoder preset that was
# verified to load and run in this environment, and both freeze settings
# wherever both are possible:
#
#   encoder              fine-tune   frozen    why
#   ngram                   yes        -       no LM to freeze
#   bert-multilingual       yes       yes
#   mpnet                   yes       yes
#   bge-large               yes       yes
#   e5-large                yes       yes
#   gte-large               yes       yes
#   xlm-roberta-large       yes       yes
#   e5-mistral-7b            -        yes      frozen_only: encode_triple pushes
#                                              3B sequences/step, so fine-tuning
#                                              7B does not fit at any useful B
#   heads: linear, mlp                         (residual_mlp also exists)
#
# BenchConfig.cell_specs() decides which combinations are real cells, so the
# impossible ones are dropped rather than needing separate config files.
# `kgfm bench run` prints the resolved cell list with each cell's settings
# before it starts — check that first if the count surprises you.
#
# Every cell runs at batch_size 256 and proj_dim 256 on purpose: B-1 is the
# number of in-batch negatives and proj_dim is the scoring width, so letting
# them vary would confound the architecture comparison with the training
# setup. 256 is the largest value the heaviest cell survives — bge-large
# fine-tuning OOMs at both 512 and 384 on a 143 GiB H200, verified by running
# the real cell on an idle GPU. See the config header for the numbers.
#
# Estimated ~15 h training + ~3 h evaluation on one H200. The two 7B cells are
# roughly a third of the training total on their own; drop them for a shorter
# run:
#   bash benchmarks/run_chembl_xlarge_compare.sh \
#       --encoders ngram,bert-multilingual,mpnet,bge-large,e5-large,gte-large,xlm-roberta-large
#
# e5-mistral-7b is a ~14 GB download on first use. Fetch it (and check HF auth)
# before committing to a day-long run:
#   python -c "from kgfm.encoders import make_encoder; make_encoder('e5-mistral-7b', freeze_encoder=True)"
#
# Resume an interrupted run — each cell skips itself when its JSON exists:
#   bash benchmarks/resume_chembl.sh latest
#
# Narrow the sweep when you only care about part of it. These are run-level
# flags, so they replace the axis rather than overriding a cell:
#   bash benchmarks/run_chembl_xlarge_compare.sh --heads linear
#   bash benchmarks/run_chembl_xlarge_compare.sh --freezes on   # frozen only
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_xlarge_compare.yaml "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
