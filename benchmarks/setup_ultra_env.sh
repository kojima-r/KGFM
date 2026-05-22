#!/usr/bin/env bash
# Build a dedicated conda env (default: kgfm-ultra) for running ULTRA.
#
# ULTRA's rspmm CUDA extension does not build / load cleanly against
# bleeding-edge PyTorch — torch 2.11 hits a `cudaErrorIllegalAddress` inside
# the sparse op. We therefore pin a known-good torch 2.5 + CUDA 12.1 combo
# in an isolated env that ``run_ultra.py`` automatically switches into.
#
# Usage:
#   bash benchmarks/setup_ultra_env.sh
#   ENV_NAME=my-ultra bash benchmarks/setup_ultra_env.sh
#   FORCE_RECREATE=1 bash benchmarks/setup_ultra_env.sh
# Note: deliberately not using `set -u`. Conda's activate/deactivate hooks
# reference unset variables (e.g. _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED in
# the gcc deactivate hook), which would abort the script under nounset.
set -eo pipefail

ENV_NAME="${ENV_NAME:-kgfm-ultra}"
PY_VERSION="${PY_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.1.1}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ultra-env] error: 'conda' not found in PATH." >&2
    exit 1
fi

# `conda activate` requires sourcing conda's shell functions in non-login shells.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

if [[ "${FORCE_RECREATE:-0}" == "1" && -d "$ENV_PREFIX" ]]; then
    echo "[ultra-env] FORCE_RECREATE=1 — removing $ENV_PREFIX"
    conda env remove -n "$ENV_NAME" -y
fi

if [[ -d "$ENV_PREFIX" ]]; then
    echo "[ultra-env] env '$ENV_NAME' already exists at $ENV_PREFIX — reusing"
else
    echo "[ultra-env] creating conda env '$ENV_NAME' (python=$PY_VERSION)"
    conda create -n "$ENV_NAME" -y "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

echo "[ultra-env] installing torch==$TORCH_VERSION ($TORCH_CUDA wheels)"
pip install --upgrade pip
pip install "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "[ultra-env] installing CUDA toolchain $CUDA_TOOLKIT_VERSION (minimal)"
# The unlabeled `nvidia` channel's solver tends to ignore pins like
# `cuda-toolkit=12.1.1` and pulls the newest cuda packages anyway. The
# `nvidia/label/cuda-<ver>` channels only contain that exact release, which
# is the only reliable way to pin the toolchain. We also install the
# components ULTRA's rspmm extension needs explicitly rather than the full
# `cuda-toolkit` metapackage (avoids unrelated tooling like cuda-nsight,
# which has had checksum issues on the labeled channel).
conda install -y -c "nvidia/label/cuda-$CUDA_TOOLKIT_VERSION" \
    cuda-nvcc \
    cuda-cudart-dev \
    cuda-libraries-dev \
    libcublas-dev \
    libcusparse-dev

# CUDA 12.1's nvcc rejects gcc>=13 by default. Pin a compatible host
# compiler in the env so torch_scatter and ULTRA's rspmm both build.
echo "[ultra-env] installing gxx_linux-64=12 host compiler"
conda install -y -c conda-forge "gxx_linux-64=12" "gcc_linux-64=12"

# conda packages the compiler as ``x86_64-conda-linux-gnu-g++`` etc.; nvcc
# invokes the host compiler through the unprefixed name ``g++`` resolved
# from PATH. Symlink the prefixed binaries so the env's gcc 12 wins over
# the system gcc 13 when run_ultra.py prepends the env bin to PATH.
ENV_BIN="$CONDA_PREFIX/bin"
for short in gcc g++ cpp; do
    target_path="$ENV_BIN/x86_64-conda-linux-gnu-${short}"
    link_path="$ENV_BIN/${short}"
    if [[ -e "$target_path" && ! -e "$link_path" ]]; then
        ln -sf "$target_path" "$link_path"
        echo "[ultra-env] symlinked $link_path -> $target_path"
    fi
done

echo "[ultra-env] installing torch_geometric + ULTRA's pip deps"
pip install torch_geometric easydict pyyaml ninja

echo "[ultra-env] building torch_scatter against the pinned torch"
pip install torch_scatter --no-build-isolation

# Activation hook so CUDA_HOME / CPATH / LIBRARY_PATH are wired up whenever
# the env is activated (covers both `conda activate kgfm-ultra` and the
# `run_ultra.py` direct-python path through CONDA_PREFIX-aware defaults).
ACTIVATE_DIR="$ENV_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/cuda-home.sh" <<'EOF'
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
EOF

# Quick sanity probe.
python - <<'PY'
import torch, torch_scatter, torch_geometric
print(f"[probe] torch={torch.__version__} cuda={torch.version.cuda} "
      f"avail={torch.cuda.is_available()}")
print(f"[probe] torch_scatter={torch_scatter.__version__} "
      f"torch_geometric={torch_geometric.__version__}")
PY

echo "[ultra-env] done. Use it via:"
echo "  conda activate $ENV_NAME"
echo "or just run benchmarks/run_ultra.py — it auto-detects this env."
