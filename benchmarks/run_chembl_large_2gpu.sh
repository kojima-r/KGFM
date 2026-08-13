#!/usr/bin/env bash
# ChEMBL large scale across two H200s. Same scale settings as
# run_chembl_large.sh (benchmarks/config_large.yaml) — only the parallelism
# differs, which is why there is no config_large_2gpu.yaml: a config file
# describes a *scale*, and GPU count is a flag.
#
# What the second GPU actually buys, and why no batch-size flag is passed:
#   in_batch_negative_loss builds its [B, B] logit matrix from the *local*
#   micro-batch — there is no all_gather of embeddings anywhere in kgfm — so
#   the negative count is the PER-DEVICE batch size, not the global one.
#   `batch_size: 512` in the config already becomes 512 per device (see
#   train._resolve_per_device_batch_size), so each rank keeps the 511
#   negatives the config was tuned for. Passing an explicit
#   --per-device-train-batch-size would be worse than passing nothing: it is a
#   single global override, so it would flatten the per-cell batch sizes that
#   `cells: <tag>: batch_size:` exists to express.
#
#   The global batch becomes 512 x 2 = 1024. Each optimizer step therefore
#   averages its gradient over twice as many examples, and 25,000 steps sees
#   25.6M examples instead of 12.8M — in roughly the same wall clock, since
#   the two ranks run concurrently.
#
#   Those extra examples are entirely *fresh*, and they are fresh in the way
#   that matters here. Each ChEMBL TSV is 10,000,000 rows and workers read
#   files sequentially, so a 25k-step run at B=512 pulls only 3.2M rows per
#   dataloader worker — it never finishes even one file. With num_workers=4
#   that means a 1-GPU run trains on ~4 of the 85 train files (32% into each);
#   2 GPUs is 8 workers, so ~8 files. Since the corpus is partitioned by
#   activity ID, each file is a different entity population, and the
#   train/valid gap is precisely a population-shift gap — so the second GPU
#   buys entity diversity, not just throughput. `max_rows_per_file` is the
#   knob that trades depth for breadth on a single GPU.
#
# So this is NOT a step-for-step replica of the 1-GPU run: same negatives and
# same time budget, twice the data per step. To match the 1-GPU training
# budget instead (~half the wall clock), halve the steps:
#       bash benchmarks/run_chembl_large_2gpu.sh --max-steps 12500
# The learning rate deliberately does NOT change. Doubling the effective batch
# usually argues for a larger step, but measured here (ngram, 1000 steps,
# global_bs=1024, valid loss / pooled MRR):
#       lr 1e-3    4.656 / 0.2013      <- default, best loss
#       lr 1.4e-3  4.754 / 0.1941
#       lr 2e-3    4.697 / 0.2070
# Non-monotonic and within 0.1 nats, with three repeats at 1e-3 coming out
# bit-identical, so there is no scaling law to ride here. Note --lr would
# override *every* cell with one value, which is wrong anyway: the default is
# per-encoder (1e-3 ngram / 3e-5 transformer). Set `cells: <tag>: lr:` in the
# config if you really want to change one.
#
# Resume an interrupted run with:  bash benchmarks/resume_chembl.sh latest
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_large.yaml \
    --nproc 2 --run-label chembl_large_2gpu "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
