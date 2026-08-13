#!/usr/bin/env bash
# ChEMBL xlarge scale across two H200s. Same scale settings as
# run_chembl_xlarge.sh (benchmarks/config_xlarge.yaml) — only the parallelism
# differs, so there is no config_xlarge_2gpu.yaml: a config file describes a
# *scale*, and GPU count is a flag.
#
# See run_chembl_large_2gpu.sh for the full reasoning. In short: the in-batch
# negatives are local to each rank (no all_gather in kgfm), so `batch_size:
# 512` stays 512 negatives per device and only the global batch doubles to
# 1024. 60,000 steps therefore sees 30.7M examples instead of 15.4M, in
# roughly the same wall clock — and because each TSV is 10M rows read
# sequentially, the second GPU doubles the number of *files* touched (~8
# workers, ~8 files of 85) rather than re-showing the same ones. The learning
# rate is unchanged; see run_chembl_large_2gpu.sh for the measurements.
#
# To keep the 1-GPU training budget instead (~half the wall clock):
#   bash benchmarks/run_chembl_xlarge_2gpu.sh --max-steps 30000
#
# This is the config most likely to be interrupted; resume with:
#   bash benchmarks/resume_chembl.sh latest
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm bench run --config benchmarks/config_xlarge.yaml \
    --nproc 2 --run-label chembl_xlarge_2gpu "$@"
#kgfm-ultra  --out-dir latest
#kgfm-motif  --out-dir latest
kgfm report --out-dir latest
