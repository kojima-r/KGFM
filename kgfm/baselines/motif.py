"""`kgfm-motif` — zero-shot MOTIF inference on the prepared ChEMBL KG.

MOTIF (https://github.com/HxyScotthuang/MOTIF) is an ULTRA-derived codebase
and shares its CLI contract, so this is the same flow with a different repo,
checkpoint, and processed-cache directory.

GPU note: MOTIF is GPU-only. Its ``HypergraphLayer`` always takes the Triton
path (``models.py`` never forwards the config's ``use_triton``), and Triton
calls ``torch.cuda.set_device``. Its ``rspmm`` extension is also JIT-compiled
against the env's torch and needs a matching nvcc. So run it in an env that
has both — ``kgfm-motif --conda-env gnn`` on this machine — rather than in a
``kgfm`` env with no CUDA toolchain.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .common import Baseline, run

SPEC = Baseline(
    method="MOTIF",
    package="motif",
    repo_url="https://github.com/HxyScotthuang/MOTIF.git",
    default_repo_dir=os.path.join("benchmarks", "MOTIF"),
    default_ckpt=os.path.join("benchmarks", "MOTIF", "ckpts", "motif_3g.pth"),
    default_config="config/transductive/MOTIF_inference.yaml",
    default_gpus="[0]",
    # Separate cache so ULTRA's and MOTIF's processed pickles can't collide.
    processed_dir="processed_motif",
    cpu_supported=False,
    cpu_unsupported_reason=(
        "its HypergraphLayer always uses the Triton kernel, which requires a "
        "CUDA device (models.py does not forward the config's use_triton)"
    ),
)


def main(argv: Optional[List[str]] = None) -> None:
    run(SPEC, argv)


if __name__ == "__main__":
    main()
