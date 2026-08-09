#!/usr/bin/env bash
# Build the conda env the ULTRA / MOTIF baselines run in: kgfm-ultra.
#
# Why a second env, when kgfm itself is one env: both baselines JIT-compile an
# `rspmm` CUDA extension against the env's torch, so they need an nvcc matching
# `torch.version.cuda` plus a host gcc that nvcc accepts. The `kgfm` env tracks
# current torch and has no CUDA toolchain, and pinning one there would drag
# kgfm's own torch along with it.
#
# MOTIF *requires* this: it is GPU-only (its HypergraphLayer always takes the
# Triton path) and cannot run in `kgfm` at all. ULTRA runs on CPU either way.
#
# The versions below are pins, not options — they are the combination known to
# build and run. Edit them here if you need to move.
#
#   bash benchmarks/setup_baseline_env.sh
#   FORCE_RECREATE=1 bash benchmarks/setup_baseline_env.sh   # rebuild from scratch
#
# Note: deliberately not using `set -u` — conda's activate/deactivate hooks
# reference unset variables and would abort the script under nounset.
set -eo pipefail

ENV_NAME=kgfm-ultra
PY_VERSION=3.11
TORCH_VERSION=2.5.1
TORCH_CUDA=cu121
CUDA_VERSION=12.1.1          # must match TORCH_CUDA

if ! command -v conda >/dev/null 2>&1; then
    echo "[baseline-env] error: 'conda' not found in PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"

if [[ "${FORCE_RECREATE:-0}" == "1" && -d "$ENV_PREFIX" ]]; then
    echo "[baseline-env] FORCE_RECREATE=1 — removing $ENV_PREFIX"
    conda env remove -n "$ENV_NAME" -y
fi

if [[ -d "$ENV_PREFIX" ]]; then
    echo "[baseline-env] env '$ENV_NAME' already exists — reusing"
else
    echo "[baseline-env] creating conda env '$ENV_NAME' (python=$PY_VERSION)"
    conda create -n "$ENV_NAME" -y "python=$PY_VERSION"
fi

conda activate "$ENV_NAME"

echo "[baseline-env] installing torch==$TORCH_VERSION ($TORCH_CUDA wheels)"
pip install --upgrade pip
pip install "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

# The unlabeled `nvidia` channel's solver ignores pins like `cuda-toolkit=12.1.1`
# and pulls the newest cuda packages anyway. The `nvidia/label/cuda-<ver>`
# channels contain only that exact release, which is the only reliable way to
# pin the toolchain. Install just the components rspmm needs rather than the
# `cuda-toolkit` metapackage (which drags in cuda-nsight, prone to checksum
# failures on the labeled channel).
echo "[baseline-env] installing CUDA $CUDA_VERSION toolchain (minimal)"
conda install -y -c "nvidia/label/cuda-$CUDA_VERSION" \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev libcublas-dev libcusparse-dev

# CUDA 12.1's nvcc rejects gcc>=13, so pin a host compiler in the env.
echo "[baseline-env] installing gcc 12 host compiler"
conda install -y -c conda-forge "gxx_linux-64=12" "gcc_linux-64=12"

# conda names them `x86_64-conda-linux-gnu-g++`; nvcc resolves the plain `g++`
# from PATH. Symlink so the env's gcc 12 wins over the system gcc.
for short in gcc g++ cpp; do
    prefixed="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-${short}"
    [[ -e "$prefixed" && ! -e "$CONDA_PREFIX/bin/${short}" ]] \
        && ln -sf "$prefixed" "$CONDA_PREFIX/bin/${short}"
done

# Point the build at the toolchain just installed. Without this, a CUDA_HOME
# inherited from the calling shell decides, and building against a CUDA that
# does not match torch fails late and cryptically.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"

echo "[baseline-env] installing torch_geometric + pip deps"
pip install torch_geometric easydict pyyaml ninja

# No prebuilt wheel matches this torch, so this compiles from source — the slow
# part. --no-cache-dir is load-bearing: pip caches wheels it built locally, so
# a torch_scatter compiled against some other torch would be reused here and
# then die at import with `undefined symbol`.
echo "[baseline-env] building torch_scatter against the pinned torch (slow)"
pip install torch_scatter --no-build-isolation --no-cache-dir

# Wire the CUDA vars up on every future `conda activate` too.
mkdir -p "$ENV_PREFIX/etc/conda/activate.d"
cat > "$ENV_PREFIX/etc/conda/activate.d/cuda-home.sh" <<'HOOK'
export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
HOOK

python - <<'PY'
import torch, torch_geometric, torch_scatter
print(f"[probe] torch={torch.__version__} cuda={torch.version.cuda} "
      f"avail={torch.cuda.is_available()}")
print(f"[probe] torch_scatter={torch_scatter.__version__} "
      f"torch_geometric={torch_geometric.__version__}")
PY

echo "[baseline-env] done. kgfm-ultra / kgfm-motif use '$ENV_NAME' by default."
