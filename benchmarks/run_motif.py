"""Zero-shot evaluation with MOTIF on the prepared chembl KG.

MOTIF (https://github.com/HxyScotthuang/MOTIF) is structurally an evolution
of ULTRA — it shares the ``util.parse_args`` contract, ``TransductiveDataset``
base class, and ``script/run.py`` entrypoint. This runner therefore mirrors
``run_ultra.py``: it injects a ``ChEMBLCustom`` dataset class into MOTIF's
``motif/datasets.py``, patches the ``torch.load`` call to bypass PyTorch
2.6+'s ``weights_only`` default, and invokes ``script/run.py`` with
``--epochs 0`` for zero-shot inference.

Run after ``benchmarks/prepare_chembl_kg.py``. The default config is the
``MOTIF_inference.yaml`` shipped with the upstream repo; pass ``--config``
to override.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


_SHIM_MARKER = "# === BEGIN benchmarks/run_motif.py injection (ChEMBLCustom) ==="
_SHIM_END = "# === END benchmarks/run_motif.py injection ==="

# Sentinel used to detect a previous injection in script/run.py.
_RUNPY_MARKER = "# === BEGIN benchmarks/run_motif.py injection (test_count) ==="
_RUNPY_END = "# === END benchmarks/run_motif.py injection (test_count) ==="


def _shim_source(data_root_abs: str) -> str:
    return f'''
{_SHIM_MARKER}
# Auto-generated; safe to delete. Re-run benchmarks/run_motif.py to recreate.
import os as _os_chembl

class ChEMBLCustom(TransductiveDataset):
    name = "chembl_custom"
    delimiter = "\\t"

    _DATA_DIR = {data_root_abs!r}

    def __init__(self, root=None, **kwargs):
        if root is None:
            root = self._DATA_DIR
        super().__init__(root=root, **kwargs)

    @property
    def raw_dir(self):
        return self._DATA_DIR

    @property
    def processed_dir(self):
        return _os_chembl.path.join(self._DATA_DIR, "processed_motif")

    def download(self):
        for fn in self.raw_file_names:
            p = _os_chembl.path.join(self.raw_dir, fn)
            if not _os_chembl.path.exists(p):
                raise FileNotFoundError(
                    f"{{p}} not found. Run benchmarks/prepare_chembl_kg.py first."
                )
{_SHIM_END}
'''


def _inject_shim(motif_dir: str, data_root_abs: str) -> None:
    path = os.path.join(motif_dir, "motif", "datasets.py")
    with open(path, "r") as f:
        content = f.read()

    # PyTorch 2.6+ `weights_only=True` default rejects MOTIF's processed
    # dataset cache. Patch the call sites idempotently.
    needle = "torch.load(self.processed_paths[0])"
    repl = "torch.load(self.processed_paths[0], weights_only=False)"
    if needle in content and repl not in content:
        content = content.replace(needle, repl)

    new_block = _shim_source(data_root_abs)

    if _SHIM_MARKER in content:
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
_COUNT_RE = re.compile(r"#test_triplets:\s*(\d+)")


def _parse_metrics(output: str) -> dict:
    found: dict = {}
    for m in _METRIC_RE.finditer(output):
        key = m.group(1).lower()
        try:
            found[key] = float(m.group(2))
        except ValueError:
            pass
    counts = _COUNT_RE.findall(output)
    if counts:
        found["n"] = int(counts[-1])
    return found


def _inject_test_count(motif_dir: str) -> None:
    """Patch ``script/run.py`` so it logs ``#test_triplets: N`` per split."""
    path = os.path.join(motif_dir, "script", "run.py")
    with open(path, "r") as f:
        content = f.read()

    while _RUNPY_MARKER in content:
        marker_at = content.index(_RUNPY_MARKER)
        start = content.rfind("\n", 0, marker_at) + 1
        end_marker = content.index(_RUNPY_END, marker_at)
        end_line = content.index("\n", end_marker) + 1
        content = content[:start] + content[end_line:]

    needle = (
        "    test_triplets = torch.cat([test_data.target_edge_index,"
        " test_data.target_edge_type.unsqueeze(0)]).t()\n"
    )
    if needle in content:
        insert = (
            f"    {_RUNPY_MARKER}\n"
            f"    if rank == 0:\n"
            f'        logger.warning("#test_triplets: %d" % len(test_triplets))\n'
            f"    {_RUNPY_END}\n"
        )
        content = content.replace(needle, needle + insert, 1)

    with open(path, "w") as f:
        f.write(content)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--motif-dir", default=os.path.join(HERE, "MOTIF"))
    p.add_argument("--data-root", default=os.path.join(HERE, "chembl_kg"))
    p.add_argument("--ckpt", default=None,
                   help="Path to a pretrained MOTIF checkpoint. "
                        "Defaults to <motif-dir>/ckpts/motif_3g.pth.")
    p.add_argument("--gpus", default="[0]")
    p.add_argument("--config",
                   default="config/transductive/MOTIF_inference.yaml",
                   help="Path to the MOTIF YAML config (relative to --motif-dir).")
    p.add_argument("--bpe", default="null")
    p.add_argument("--out", default=os.path.join(HERE, "results", "motif.json"))
    args = p.parse_args()

    if not os.path.isdir(args.motif_dir):
        sys.exit(
            f"MOTIF repo not found at {args.motif_dir}. "
            f"Run `bash benchmarks/setup.sh` first."
        )
    if not os.path.isdir(args.data_root):
        sys.exit(
            f"Prepared chembl KG not found at {args.data_root}. "
            f"Run `python benchmarks/prepare_chembl_kg.py` first."
        )

    ckpt = args.ckpt or os.path.join(args.motif_dir, "ckpts", "motif_3g.pth")
    if not os.path.isfile(ckpt):
        sys.exit(
            f"MOTIF checkpoint not found at {ckpt}. "
            f"Pass --ckpt or fetch one into <motif-dir>/ckpts/."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data_root_abs = os.path.abspath(args.data_root)
    print(f"[motif] injecting ChEMBLCustom into {args.motif_dir}/motif/datasets.py")
    _inject_shim(args.motif_dir, data_root_abs)
    print(f"[motif] injecting test-count logger into {args.motif_dir}/script/run.py")
    _inject_test_count(args.motif_dir)

    cmd = [
        sys.executable, "script/run.py",
        "-c", args.config,
        "--dataset", "ChEMBLCustom",
        "--gpus", args.gpus,
        "--epochs", "0",
        "--bpe", args.bpe,
        "--ckpt", os.path.abspath(ckpt),
    ]
    print(f"[motif] running: {' '.join(cmd)}  (cwd={args.motif_dir})")

    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=args.motif_dir, text=True, capture_output=True,
    )
    elapsed = time.time() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        out = {
            "method": "MOTIF",
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
        "method": "MOTIF",
        "mode": "zero-shot",
        "ckpt": ckpt,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
        "data_root": data_root_abs,
        "config": args.config,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[motif] wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
