#!/usr/bin/env bash
# Clone ULTRA + MOTIF and check their conda env can import what they need.
# Idempotent. Each baseline owns its own setup; this just runs both.
#
# Checkpoints ship with the clones, so downloading one is rarely needed:
#   kgfm-ultra --fetch-ckpt
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

kgfm-ultra --setup
kgfm-motif --setup
