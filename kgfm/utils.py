import hashlib
import os
import subprocess
from typing import List, Optional, Tuple

import torch


def _visible_devices() -> "Optional[List[int]]":
    """Physical GPU ids torch can see, in torch's own ordinal order.

    `CUDA_VISIBLE_DEVICES=1` makes physical GPU 1 into `cuda:0`, and there is
    no `cuda:1` at all. nvidia-smi keeps reporting physical ids either way, so
    anything that picks a device from nvidia-smi output has to translate.
    Returns None when the variable is unset (every device visible, ordinals =
    physical ids) and [] when it is set to empty (CUDA disabled).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            # UUID form (GPU-xxxx). We cannot map those to nvidia-smi indices,
            # so decline to guess and let the caller fall back to cuda:0.
            return None
    return ids


def pick_free_gpu(min_free_mib: int = 4096) -> torch.device:
    """Return the CUDA device with the most free memory (and lowest utilization).

    Falls back to CPU if no GPU has at least `min_free_mib` free memory.

    The index returned is a **torch ordinal**, not the physical id nvidia-smi
    reports. The two differ whenever CUDA_VISIBLE_DEVICES pins the process to
    a subset: with `CUDA_VISIBLE_DEVICES=1` there is exactly one device and it
    is `cuda:0`, so returning nvidia-smi's `1` raised
    `CUDA error: invalid device ordinal` — which is how any job pinned to a
    non-first GPU used to die before its first forward.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return torch.device("cuda:0")
    visible = _visible_devices()
    # physical id -> torch ordinal, or identity when the variable is unset.
    ordinal = ({phys: i for i, phys in enumerate(visible)}
               if visible is not None else None)
    candidates: List[Tuple[int, int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free_mib, util = int(parts[0]), int(parts[1]), int(parts[2])
        if ordinal is not None:
            if idx not in ordinal:
                continue
            idx = ordinal[idx]
        candidates.append((idx, free_mib, util))
    # Whatever nvidia-smi said, torch is the authority on how many devices
    # exist; a mismatch (MIG, a driver that renumbers) must not produce an
    # out-of-range ordinal.
    candidates = [c for c in candidates if c[0] < torch.cuda.device_count()]
    if not candidates:
        return torch.device("cuda:0")
    # Prefer most free memory, then lowest utilization.
    candidates.sort(key=lambda c: (-c[1], c[2]))
    idx, free_mib, util = candidates[0]
    if free_mib < min_free_mib:
        return torch.device("cpu")
    return torch.device(f"cuda:{idx}")


def file_split_bucket(path: str, n_buckets: int = 10) -> int:
    """Deterministic bucket index in [0, n_buckets) based on filename hash."""
    h = hashlib.blake2b(path.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % n_buckets


def is_test_file(path: str, test_buckets: int = 1, n_buckets: int = 10) -> bool:
    """Default split: 10% of files are test (bucket 0)."""
    return file_split_bucket(path, n_buckets) < test_buckets
