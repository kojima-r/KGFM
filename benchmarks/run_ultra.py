"""Zero-shot evaluation with ULTRA on the prepared chembl KG.

ULTRA is a foundation model for KG reasoning. We do *not* fine-tune; we use
one of the released pretrained checkpoints (default ``ultra_50g.pth``) and
run inference (``--epochs 0``) on the chembl test split prepared by
``benchmarks/prepare_chembl_kg.py``.

Environment switching
---------------------
ULTRA's CUDA extension does not run cleanly under bleeding-edge PyTorch.
This runner therefore defaults to an isolated conda env (``kgfm-ultra``,
created by ``benchmarks/setup_ultra_env.sh``) and invokes that env's Python
binary as a subprocess. The current shell stays on whatever env you use
for kgfm itself.

Override with ``--env <name>`` or ``--python /abs/path/to/python``.

How it works
------------
ULTRA looks up datasets via ``getattr(ultra.datasets, <ClassName>)``. We
therefore *append* a small ``ChEMBLCustom`` class to
``ULTRA/ultra/datasets.py`` (idempotently, guarded by a sentinel comment)
that subclasses ``TransductiveDataset`` and points at our prepared
``train.txt`` / ``valid.txt`` / ``test.txt``.

ULTRA's ``ultra/util.parse_args`` discovers ``{{ var }}`` placeholders in
the YAML config and turns them into required CLI flags, so we invoke
``script/run.py`` with the same set of flags ULTRA's own examples use:
``--dataset --gpus --epochs --bpe --ckpt``.

If upstream ULTRA renames ``TransductiveDataset`` or moves things, edit
the dataset shim template at the top of this file. The subprocess
invocation contract is otherwise stable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_NAME = "kgfm-ultra"


# Sentinel used to detect a previous injection in ultra/datasets.py.
_SHIM_MARKER = "# === BEGIN benchmarks/run_ultra.py injection (ChEMBLCustom) ==="
_SHIM_END = "# === END benchmarks/run_ultra.py injection ==="


def _shim_source(data_root_abs: str) -> str:
    return f'''
{_SHIM_MARKER}
# Auto-generated; safe to delete. Re-run benchmarks/run_ultra.py to recreate.
import os as _os_chembl

class ChEMBLCustom(TransductiveDataset):
    name = "chembl_custom"
    # MUST be a literal tab — the prepared TSVs may contain entities whose
    # surface form has internal spaces (e.g. RDF literal strings), so the
    # parent's default ``l.split()`` would mis-tokenize those rows.
    delimiter = "\t"

    _DATA_DIR = {data_root_abs!r}

    def __init__(self, root=None, **kwargs):
        # We accept a 'root' for compatibility but always serve files
        # from _DATA_DIR. Each call still needs *some* root for
        # InMemoryDataset's bookkeeping (the processed cache lives there).
        if root is None:
            root = self._DATA_DIR
        super().__init__(root=root, **kwargs)

    @property
    def raw_dir(self):
        return self._DATA_DIR

    @property
    def processed_dir(self):
        return _os_chembl.path.join(self._DATA_DIR, "processed")

    def download(self):
        for fn in self.raw_file_names:
            p = _os_chembl.path.join(self.raw_dir, fn)
            if not _os_chembl.path.exists(p):
                raise FileNotFoundError(
                    f"{{p}} not found. Run benchmarks/prepare_chembl_kg.py first."
                )
{_SHIM_END}
'''


def _inject_shim(ultra_dir: str, data_root_abs: str) -> None:
    path = os.path.join(ultra_dir, "ultra", "datasets.py")
    with open(path, "r") as f:
        content = f.read()

    # PyTorch 2.6+ flipped torch.load's `weights_only` default to True, which
    # rejects ULTRA's processed dataset pickles. Patch the call sites
    # idempotently before injecting the shim.
    needle = "torch.load(self.processed_paths[0])"
    repl = "torch.load(self.processed_paths[0], weights_only=False)"
    if needle in content and repl not in content:
        content = content.replace(needle, repl)

    new_block = _shim_source(data_root_abs)

    if _SHIM_MARKER in content:
        # Strip any prior injection so we can refresh _DATA_DIR.
        start = content.index(_SHIM_MARKER)
        end = content.index(_SHIM_END, start) + len(_SHIM_END)
        content = content[:start].rstrip() + "\n" + content[end:].lstrip()

    if not content.endswith("\n"):
        content += "\n"
    content += new_block

    with open(path, "w") as f:
        f.write(content)


_METRIC_RE = re.compile(
    r"(mrr|hits@1|hits@3|hits@10|mr)\s*[:=]\s*([0-9eE+\-.]+)",
    re.IGNORECASE,
)


def _parse_metrics(output: str) -> dict:
    """Pick the *last* occurrence of each metric in ULTRA's stdout/stderr."""
    found: dict = {}
    for m in _METRIC_RE.finditer(output):
        key = m.group(1).lower()
        try:
            found[key] = float(m.group(2))
        except ValueError:
            pass
    return found


def _conda_base() -> Optional[str]:
    """Return ``$CONDA_BASE`` via ``conda info --base``, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["conda", "info", "--base"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _resolve_env_python(env_name: str) -> Optional[str]:
    """Find ``<conda-base>/envs/<env_name>/bin/python`` if it exists."""
    base = _conda_base()
    if not base:
        return None
    candidate = os.path.join(base, "envs", env_name, "bin", "python")
    return candidate if os.path.isfile(candidate) else None


def _env_with_cuda_home(env_python: str) -> dict:
    """Mirror the activation hook installed by setup_ultra_env.sh.

    ULTRA's JIT-compiled rspmm extension needs CUDA_HOME / CPATH /
    LIBRARY_PATH pointing at the conda env's CUDA toolkit. Subprocess
    invocations don't go through ``conda activate``, so we set them by hand.
    """
    env = os.environ.copy()
    env_root = os.path.dirname(os.path.dirname(env_python))  # .../envs/<name>/
    env["CUDA_HOME"] = env_root
    env_bin = os.path.join(env_root, "bin")
    # Make sure nvcc finds the env's gcc 12 (symlinked as g++) BEFORE the
    # system gcc 13, which CUDA 12.1's nvcc refuses to use.
    env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
    targets_inc = os.path.join(env_root, "targets", "x86_64-linux", "include")
    targets_lib = os.path.join(env_root, "targets", "x86_64-linux", "lib")
    if os.path.isdir(targets_inc):
        env["CPATH"] = targets_inc + os.pathsep + env.get("CPATH", "")
    if os.path.isdir(targets_lib):
        env["LIBRARY_PATH"] = (
            targets_lib + os.pathsep + env.get("LIBRARY_PATH", "")
        )
    return env


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ultra-dir", default=os.path.join(HERE, "ULTRA"))
    p.add_argument("--data-root", default=os.path.join(HERE, "chembl_kg"))
    p.add_argument("--ckpt", default=None,
                   help="Path to a pretrained ULTRA checkpoint. "
                        "Defaults to <ultra-dir>/ckpts/ultra_50g.pth.")
    p.add_argument("--gpus", default="[0]",
                   help="JSON list passed straight through to ULTRA "
                        "(e.g. '[0]' or '[1]').")
    p.add_argument("--config",
                   default="config/transductive/inference.yaml",
                   help="Path to the ULTRA YAML config (relative to --ultra-dir).")
    p.add_argument("--bpe", default="null",
                   help="batch_per_epoch ULTRA variable; 'null' = use full data.")
    p.add_argument("--env", default=DEFAULT_ENV_NAME,
                   help="Conda env name to invoke ULTRA from "
                        f"(default: {DEFAULT_ENV_NAME}). Build it with "
                        f"`bash benchmarks/setup_ultra_env.sh`.")
    p.add_argument("--python", default=None,
                   help="Absolute path to the Python interpreter to run ULTRA "
                        "with. Overrides --env.")
    p.add_argument("--out", default=os.path.join(HERE, "results", "ultra.json"))
    p.add_argument("--cuda-launch-blocking", action="store_true",
                   help="Set CUDA_LAUNCH_BLOCKING=1 in the subprocess for "
                        "synchronous CUDA error reporting (debugging only).")
    args = p.parse_args()

    if not os.path.isdir(args.ultra_dir):
        sys.exit(
            f"ULTRA repo not found at {args.ultra_dir}. "
            f"Run `bash benchmarks/setup.sh` first."
        )
    if not os.path.isdir(args.data_root):
        sys.exit(
            f"Prepared chembl KG not found at {args.data_root}. "
            f"Run `python benchmarks/prepare_chembl_kg.py` first."
        )

    ckpt = args.ckpt or os.path.join(args.ultra_dir, "ckpts", "ultra_50g.pth")
    if not os.path.isfile(ckpt):
        sys.exit(
            f"ULTRA checkpoint not found at {ckpt}. "
            f"Pass --ckpt or fetch one into <ultra-dir>/ckpts/."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Resolve which Python to run ULTRA with. Prefer the dedicated env so
    # that ULTRA's pinned torch / CUDA toolchain doesn't clash with kgfm's.
    if args.python:
        env_python = args.python
        env_label = args.python
        if not os.path.isfile(env_python):
            sys.exit(f"--python path does not exist: {env_python}")
    else:
        env_python = _resolve_env_python(args.env)
        env_label = f"conda env '{args.env}'"
        if env_python is None:
            sys.exit(
                f"Conda env '{args.env}' not found. Build it with "
                f"`bash benchmarks/setup_ultra_env.sh` (or pass "
                f"--python <interpreter> to use a manually-prepared one)."
            )

    subprocess_env = _env_with_cuda_home(env_python)
    if args.cuda_launch_blocking:
        subprocess_env["CUDA_LAUNCH_BLOCKING"] = "1"

    data_root_abs = os.path.abspath(args.data_root)
    print(f"[ultra] injecting ChEMBLCustom into {args.ultra_dir}/ultra/datasets.py")
    _inject_shim(args.ultra_dir, data_root_abs)

    cmd = [
        env_python, "script/run.py",
        "-c", args.config,
        "--dataset", "ChEMBLCustom",
        "--gpus", args.gpus,
        "--epochs", "0",
        "--bpe", args.bpe,
        "--ckpt", os.path.abspath(ckpt),
    ]
    print(f"[ultra] python={env_python}  ({env_label})")
    print(f"[ultra] CUDA_HOME={subprocess_env.get('CUDA_HOME', '<unset>')}")
    print(f"[ultra] running: {' '.join(cmd)}  (cwd={args.ultra_dir})")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=args.ultra_dir, env=subprocess_env,
        text=True, capture_output=True,
    )
    elapsed = time.time() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        # Persist a failure record so aggregate.py can surface it.
        out = {
            "method": "ULTRA",
            "mode": "zero-shot",
            "ckpt": ckpt,
            "elapsed_seconds": round(elapsed, 1),
            "metrics": {},
            "error": f"non-zero exit code {proc.returncode}",
            "command": cmd,
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        sys.exit(proc.returncode)

    metrics = _parse_metrics(proc.stdout + "\n" + proc.stderr)
    out = {
        "method": "ULTRA",
        "mode": "zero-shot",
        "ckpt": ckpt,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "data_root": data_root_abs,
        "config": args.config,
        "python": env_python,
        "env": args.env if not args.python else None,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ultra] wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
