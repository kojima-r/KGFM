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
# Why these BERT-specific defaults:
#   --transformer-batch-size 64 — BERT-base (177M params, 12 layers, hidden=768)
#     full fine-tune at L=128 cannot fit B=1024 on a single H200 (140GB):
#     encode_triple bundles 3 sequences per example, so B=1024 means a
#     3072×128 transformer forward whose activations alone OOM. B=64 gives
#     a 192×128 forward that comfortably fits with the AdamW state and bf16.
#   --proj-dim 256 — frozen-BERT cells need a learnable head to train at all
#     (with proj_dim=None and a frozen LM, the optimizer has zero trainable
#     parameters). 256 doubles as a no-op for the ngram encoder, which is
#     already 256-dim so the projection collapses to nn.Identity.
#   --kgfm-freezes off,on — sweep the same encoder both fully fine-tuned and
#     as a frozen feature extractor + trainable projection head.
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
    --max-train               500000 \
    --max-valid                10000 \
    --max-test                 10000 \
    --max-steps                 2000 \
    --batch-size                1024 \
    --transformer-batch-size      64 \
    --proj-dim                   256 \
    --kgfm-freezes            off,on \
    --max-filter-tails        200000 \
    --max-filter-rows        5000000 \
    "$@"
