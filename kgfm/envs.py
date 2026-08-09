"""Conda-env resolution for anything that spawns a child process.

Everything runs under a single conda env, ``kgfm`` by default. We resolve
``<conda base>/envs/<name>/bin/python`` explicitly instead of trusting the
interpreter that happens to be running, so a command behaves the same whether
it was launched from cron, a bare shell, or an activated env — and so the
torch version recorded in meta.json is the one the work actually used.

Only child processes need this: work that stays in-process obviously runs
under whatever imported it. It matters for the kgfm sweep (each cell is a
fresh process, optionally under torchrun) and for `kgfm-ultra` / `kgfm-motif`,
which invoke their upstream ``script/run.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from typing import List, Optional, Sequence

DEFAULT_ENV = "kgfm"


@lru_cache(maxsize=1)
def conda_base() -> Optional[str]:
    """Root of the conda installation, or None when conda isn't available."""
    try:
        out = subprocess.check_output(
            ["conda", "info", "--base"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def env_prefix(env_name: str = DEFAULT_ENV) -> Optional[str]:
    base = conda_base()
    if not base:
        return None
    prefix = os.path.join(base, "envs", env_name)
    return prefix if os.path.isdir(prefix) else None


def env_python(env_name: str = DEFAULT_ENV) -> str:
    """Interpreter for ``env_name``; falls back to the current one."""
    prefix = env_prefix(env_name)
    if prefix:
        candidate = os.path.join(prefix, "bin", "python")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def kgfm_cmd(env_name: str = DEFAULT_ENV) -> List[str]:
    """Command prefix that re-enters this CLI in ``env_name``."""
    return [env_python(env_name), "-m", "kgfm"]


def torchrun_cmd(env_name: str = DEFAULT_ENV) -> List[str]:
    """Command prefix for torchrun in ``env_name``.

    Prefers the console script; ``python -m torch.distributed.run`` is the
    same launcher and covers envs where the script wasn't installed.
    """
    python = env_python(env_name)
    candidate = os.path.join(os.path.dirname(python), "torchrun")
    if os.path.isfile(candidate):
        return [candidate]
    return [python, "-m", "torch.distributed.run"]


def missing_modules(names: Sequence[str], env_name: str = DEFAULT_ENV) -> List[str]:
    """Which of ``names`` ``env_name`` cannot import."""
    probe = (
        "import importlib\n"
        f"names = {list(names)!r}\n"
        "missing = []\n"
        "for n in names:\n"
        "    try:\n"
        "        importlib.import_module(n)\n"
        "    except Exception:\n"
        "        missing.append(n)\n"
        "print(','.join(missing))\n"
    )
    try:
        out = subprocess.check_output(
            [env_python(env_name), "-c", probe], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return list(names)
    return [name for name in out.split(",") if name]


def describe(env_name: str = DEFAULT_ENV) -> dict:
    """Python / torch versions of ``env_name``, for meta.json."""
    probe = (
        "import platform, json\n"
        "info = {'python': 'Python ' + platform.python_version(), 'torch': 'n/a'}\n"
        "try:\n"
        "    import torch\n"
        "    info['torch'] = torch.__version__\n"
        "except Exception:\n"
        "    pass\n"
        "print(json.dumps(info))\n"
    )
    try:
        out = subprocess.check_output(
            [env_python(env_name), "-c", probe], text=True, stderr=subprocess.DEVNULL
        )
        import json

        return json.loads(out)
    except Exception:                                   # noqa: BLE001
        return {"python": "n/a", "torch": "n/a"}


def torch_cuda_version(env_name: str = DEFAULT_ENV) -> Optional[str]:
    """``torch.version.cuda`` in ``env_name`` (None for a CPU-only build)."""
    try:
        out = subprocess.check_output(
            [env_python(env_name), "-c",
             "import torch; print(torch.version.cuda or '')"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def _nvcc_matches_torch(cuda_home: str, env_name: str) -> bool:
    """True when ``cuda_home``'s nvcc has the same CUDA version as torch."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return False
    try:
        out = subprocess.check_output(
            [nvcc, "--version"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    import re

    match = re.search(r"release (\d+\.\d+)", out)
    return bool(match) and match.group(1) == torch_cuda_version(env_name)


def cuda_build_env(env_name: str = DEFAULT_ENV) -> dict:
    """Environment for a child that JIT-builds a CUDA extension.

    ULTRA / MOTIF compile ``rspmm`` at import time via torch's cpp_extension,
    which needs CUDA_HOME and a host compiler nvcc accepts. Each variable is
    only set when the env really provides it — pointing CUDA_HOME at a prefix
    without nvcc makes the build fail in a much more confusing way than not
    setting it at all (and CPU inference doesn't need any of this).
    """
    env = os.environ.copy()

    # An inherited CUDA_HOME is a liability, not a help: a shell that once
    # activated another conda env still exports its prefix, and building
    # against a CUDA that doesn't match torch's fails late and cryptically
    # ("libcudart.so.12: cannot open shared object file" at extension load).
    # Drop it unless its nvcc matches torch.
    inherited = env.get("CUDA_HOME")
    if inherited and not _nvcc_matches_torch(inherited, env_name):
        for var in ("CUDA_HOME", "CUDA_PATH"):
            env.pop(var, None)

    prefix = env_prefix(env_name)
    if not prefix:
        return env
    env_bin = os.path.join(prefix, "bin")
    if os.path.isfile(os.path.join(env_bin, "nvcc")):
        env["CUDA_HOME"] = prefix
    # Put the env's bin first so nvcc resolves the env's gcc rather than a
    # system one it may refuse to work with.
    env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
    targets_inc = os.path.join(prefix, "targets", "x86_64-linux", "include")
    targets_lib = os.path.join(prefix, "targets", "x86_64-linux", "lib")
    if os.path.isdir(targets_inc):
        env["CPATH"] = targets_inc + os.pathsep + env.get("CPATH", "")
    if os.path.isdir(targets_lib):
        env["LIBRARY_PATH"] = targets_lib + os.pathsep + env.get("LIBRARY_PATH", "")
    return env
