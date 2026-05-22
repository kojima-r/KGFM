"""Streaming data pipeline.

Each TSV row is expected to have 6 tab-separated columns:
    ノード１タイプ \t 関係タイプ \t ノード２タイプ \t ノード１テキスト \t 関係テキスト \t ノード２テキスト

Files are huge and numerous (~12 TB across 19k files), so:
- We never load whole files into memory; we stream line-by-line.
- Train/test split is done at the FILE level via a deterministic hash bucket.
- Workers shard files among themselves (each worker only opens its own files).
"""

from __future__ import annotations

import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

from .utils import file_split_bucket, is_test_file


csv.field_size_limit(2**31 - 1)


@dataclass
class Triple:
    n1_type: str
    rel_type: str
    n2_type: str
    n1_text: str
    rel_text: str
    n2_text: str


def discover_tsv_files(root: str = "data", pattern: str = "**/latest/*.tsv") -> List[str]:
    """Find all TSV files under data/**/latest/*.tsv (follows symlinks)."""
    pat = os.path.join(root, pattern)
    files = sorted(glob.glob(pat, recursive=True))
    return files


def read_file_list(path: str) -> List[str]:
    """Read a text file with one TSV path per line.

    - Blank lines and lines starting with '#' are skipped.
    - Paths may be absolute or relative to the current working directory.
    - Whitespace is stripped.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def split_files_three_way(
    files: Sequence[str],
    valid_buckets: int = 1,
    test_buckets: int = 1,
    n_buckets: int = 10,
) -> Tuple[List[str], List[str], List[str]]:
    """Hash-bucket fallback split when no file lists are provided.

    Default 80/10/10: bucket 0 -> test, bucket 1 -> valid, rest -> train.
    """
    train, valid, test = [], [], []
    for f in files:
        b = file_split_bucket(f, n_buckets)
        if b < test_buckets:
            test.append(f)
        elif b < test_buckets + valid_buckets:
            valid.append(f)
        else:
            train.append(f)
    return train, valid, test


def split_files(
    files: Sequence[str], test_buckets: int = 1, n_buckets: int = 10
) -> Tuple[List[str], List[str]]:
    """Backwards-compatible 2-way split (kept for existing callers)."""
    train, test = [], []
    for f in files:
        (test if is_test_file(f, test_buckets, n_buckets) else train).append(f)
    return train, test


def _iter_tsv_rows(path: str, max_text_len: int = 512) -> Iterator[Triple]:
    """Yield rows from a TSV. Skips malformed rows. Truncates long texts."""
    try:
        f = open(path, "r", encoding="utf-8", errors="replace", newline="")
    except OSError:
        return
    with f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) < 6:
                continue
            n1t, rt, n2t, n1, rel, n2 = row[0], row[1], row[2], row[3], row[4], row[5]
            if not n1 or not rel or not n2:
                continue
            yield Triple(
                n1_type=n1t,
                rel_type=rt,
                n2_type=n2t,
                n1_text=n1[:max_text_len],
                rel_text=rel[:max_text_len],
                n2_text=n2[:max_text_len],
            )


class StreamingTripleDataset(IterableDataset):
    """Streams triples from a list of TSV files.

    - Workers shard files (worker i takes files where idx % num_workers == i).
    - Optional shuffling: shuffles the FILE list per epoch and uses a small
      in-memory shuffle buffer over rows.
    - Optional row subsampling: yields each row with probability `row_keep_prob`
      to bound the work per epoch over multi-TB data.
    """

    def __init__(
        self,
        files: Sequence[str],
        shuffle_files: bool = True,
        shuffle_buffer: int = 8192,
        row_keep_prob: float = 1.0,
        max_text_len: int = 512,
        seed: int = 0,
        max_rows_per_file: Optional[int] = None,
    ):
        super().__init__()
        self.files = list(files)
        self.shuffle_files = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.row_keep_prob = row_keep_prob
        self.max_text_len = max_text_len
        self.seed = seed
        self.max_rows_per_file = max_rows_per_file

    def _worker_files(self) -> List[str]:
        info = get_worker_info()
        files = list(self.files)
        if self.shuffle_files:
            rng = random.Random(self.seed + (info.id if info else 0))
            rng.shuffle(files)
        if info is None:
            return files
        return [f for i, f in enumerate(files) if i % info.num_workers == info.id]

    def __iter__(self) -> Iterator[Triple]:
        info = get_worker_info()
        worker_seed = self.seed + (info.id if info else 0)
        rng = random.Random(worker_seed)
        files = self._worker_files()

        buffer: List[Triple] = []

        def emit(t: Triple) -> Iterator[Triple]:
            if self.shuffle_buffer <= 1:
                yield t
                return
            if len(buffer) < self.shuffle_buffer:
                buffer.append(t)
                return
            j = rng.randrange(self.shuffle_buffer)
            yield buffer[j]
            buffer[j] = t

        for path in files:
            n_emitted = 0
            for tri in _iter_tsv_rows(path, self.max_text_len):
                if self.row_keep_prob < 1.0 and rng.random() >= self.row_keep_prob:
                    continue
                for out in emit(tri):
                    yield out
                n_emitted += 1
                if (
                    self.max_rows_per_file is not None
                    and n_emitted >= self.max_rows_per_file
                ):
                    break

        # Drain the shuffle buffer.
        rng.shuffle(buffer)
        for t in buffer:
            yield t


def collate_triples(batch: Sequence[Triple]) -> dict:
    """Collate triples into raw text lists (encoding happens inside the model)."""
    return {
        "h_text": [t.n1_text for t in batch],
        "r_text": [t.rel_text for t in batch],
        "t_text": [t.n2_text for t in batch],
        "h_type": [t.n1_type for t in batch],
        "r_type": [t.rel_type for t in batch],
        "t_type": [t.n2_type for t in batch],
    }
