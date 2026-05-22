import hashlib
import os
import subprocess
from typing import List, Tuple

import torch


def pick_free_gpu(min_free_mib: int = 4096) -> torch.device:
    """Return the CUDA device with the most free memory (and lowest utilization).
    Falls back to CPU if no GPU has at least `min_free_mib` free memory."""
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
    candidates: List[Tuple[int, int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free_mib, util = int(parts[0]), int(parts[1]), int(parts[2])
        candidates.append((idx, free_mib, util))
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
