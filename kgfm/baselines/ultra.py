"""`kgfm-ultra` — zero-shot ULTRA inference on the prepared ChEMBL KG.

ULTRA (https://github.com/DeepGraphLearning/ULTRA) is a KG foundation model
with released checkpoints, so there is no training here: we run its own
``script/run.py`` with ``--epochs 0``.

GPU note: on H200 (sm_90) ULTRA's ``rspmm`` CUDA kernel dies with
``cudaErrorIllegalAddress``. It reproduces on stock ULTRA datasets and under a
torch-2.5.1-pinned env, so it is an upstream bug, not a local
misconfiguration — hence the CPU default (``--gpus null``).
"""

from __future__ import annotations

import os
from typing import List, Optional

from .common import Baseline, run

SPEC = Baseline(
    method="ULTRA",
    package="ultra",
    repo_url="https://github.com/DeepGraphLearning/ULTRA.git",
    default_repo_dir=os.path.join("benchmarks", "ULTRA"),
    default_ckpt=os.path.join("benchmarks", "ULTRA", "ckpts", "ultra_50g.pth"),
    default_config="config/transductive/inference.yaml",
    default_gpus="null",
    processed_dir="processed",
    ckpt_url=(
        "https://github.com/DeepGraphLearning/ULTRA/releases/download/v1.1.0/"
        "ultra_50g.pth"
    ),
)


def main(argv: Optional[List[str]] = None) -> None:
    run(SPEC, argv)


if __name__ == "__main__":
    main()
