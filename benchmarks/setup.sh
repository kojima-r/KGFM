#!/usr/bin/env bash
# Clone ULTRA and MOTIF into benchmarks/ and (optionally) install their deps.
#
# Usage:
#   bash benchmarks/setup.sh                  # clone only
#   INSTALL_DEPS=1 bash benchmarks/setup.sh   # also pip-install requirements
#   FETCH_CKPTS=1  bash benchmarks/setup.sh   # also fetch ULTRA pretrained ckpts
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

clone_or_update() {
    local url="$1"
    local dir="$2"
    if [[ -d "$dir/.git" ]]; then
        echo "[setup] $dir already cloned — pulling latest"
        git -C "$dir" pull --ff-only || echo "[setup] (warn) pull failed for $dir"
    else
        echo "[setup] cloning $url -> $dir"
        git clone --depth 1 "$url" "$dir"
    fi
}

clone_or_update https://github.com/DeepGraphLearning/ULTRA.git ULTRA
clone_or_update https://github.com/HxyScotthuang/MOTIF.git    MOTIF

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    for d in ULTRA MOTIF; do
        if [[ -f "$d/requirements.txt" ]]; then
            echo "[setup] pip install -r $d/requirements.txt"
            pip install -r "$d/requirements.txt"
        fi
    done
fi

if [[ "${FETCH_CKPTS:-0}" == "1" ]]; then
    mkdir -p ULTRA/ckpts
    if [[ ! -f ULTRA/ckpts/ultra_50g.pth ]]; then
        echo "[setup] fetching ultra_50g.pth"
        # ULTRA hosts checkpoints on its own GitHub release page.
        # Update the URL if it changes upstream.
        curl -L -o ULTRA/ckpts/ultra_50g.pth \
            https://github.com/DeepGraphLearning/ULTRA/releases/download/v1.1.0/ultra_50g.pth \
            || echo "[setup] (warn) ckpt download failed; fetch manually."
    fi
fi

echo "[setup] done."
