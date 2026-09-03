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

    **Files are interleaved, not concatenated** (`interleave_files`, on by
    default). Reading each file to its end before opening the next makes the
    stream a sequence of distinct distributions rather than a sample of one:
    ChEMBL is partitioned by activity ID, so files differ in relation type and
    text length (measured: mean tail length 26.6 to 53.9 characters across
    files), and the 16k shuffle buffer is far too small to mix across an 8M-row
    file. Trained that way, a model specialises to whichever block it is
    currently reading — observed directly as validation loss swinging between
    0.36 and 0.79 MRR as the loader crossed file boundaries, and as a
    monotonically rising validation loss over the second half of a full-epoch
    run. Round-robin over the worker's files puts several populations in every
    batch instead, at the cost of holding one open handle per file.
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
        interleave_files: bool = True,
        interleave_chunk: int = 64,
    ):
        super().__init__()
        self.files = list(files)
        self.shuffle_files = shuffle_files
        self.shuffle_buffer = shuffle_buffer
        self.row_keep_prob = row_keep_prob
        self.max_text_len = max_text_len
        self.seed = seed
        self.max_rows_per_file = max_rows_per_file
        self.interleave_files = interleave_files
        # Rows taken from one file before moving to the next. 1 would mix
        # maximally but turn a sequential scan into random access; a chunk
        # keeps the reads sequential while still putting ~num_files distinct
        # populations into every batch of a few hundred.
        self.interleave_chunk = max(1, int(interleave_chunk))

    def _worker_files(self) -> List[str]:
        info = get_worker_info()
        files = list(self.files)
        if self.shuffle_files:
            rng = random.Random(self.seed + (info.id if info else 0))
            rng.shuffle(files)
        if info is None:
            return files
        return [f for i, f in enumerate(files) if i % info.num_workers == info.id]

    def _rows(self, files: List[str], rng: random.Random) -> Iterator[Triple]:
        """Rows from ``files``, interleaved or concatenated, with the caps applied."""
        if not self.interleave_files:
            for path in files:
                n = 0
                for tri in _iter_tsv_rows(path, self.max_text_len):
                    if self.row_keep_prob < 1.0 and rng.random() >= self.row_keep_prob:
                        continue
                    yield tri
                    n += 1
                    if self.max_rows_per_file is not None and n >= self.max_rows_per_file:
                        break
            return

        # Round-robin. Each file keeps its own iterator and row budget; a file
        # that runs out (or hits max_rows_per_file) drops out of the rotation
        # and the rest carry on, so an unequal file list degrades to the
        # concatenated behaviour at the tail rather than stopping early.
        streams = [_iter_tsv_rows(p, self.max_text_len) for p in files]
        counts = [0] * len(streams)
        alive = list(range(len(streams)))
        while alive:
            still: List[int] = []
            for i in alive:
                taken = 0
                while taken < self.interleave_chunk:
                    if (self.max_rows_per_file is not None
                            and counts[i] >= self.max_rows_per_file):
                        break
                    try:
                        tri = next(streams[i])
                    except StopIteration:
                        break
                    counts[i] += 1
                    taken += 1
                    if self.row_keep_prob < 1.0 and rng.random() >= self.row_keep_prob:
                        continue
                    yield tri
                else:
                    still.append(i)          # budget left; keep it in rotation
            alive = still

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

        for tri in self._rows(files, rng):
            for out in emit(tri):
                yield out

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


# ---------------------------------------------------------------------------
# Row counting, for `--max-epoch`
# ---------------------------------------------------------------------------
#
# "One epoch" is only meaningful if the size of the corpus is known, and these
# TSVs are far too large to infer it from: the ChEMBL files are not uniform
# (1.81 GB down to 39 KB), so scaling one file's row count by the file count is
# off by 26%. So they get counted — 105 GiB reads at ~1.3 GB/s, about a minute
# — and the result is cached by (path, size, mtime) so only the first run pays.

_ROWCOUNT_CACHE_ENV = "KGFM_ROWCOUNT_CACHE"


def _rowcount_cache_path() -> str:
    override = os.environ.get(_ROWCOUNT_CACHE_ENV)
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "kgfm", "rowcounts.json")


def _load_rowcount_cache(path: str) -> dict:
    try:
        import json

        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_rowcount_cache(path: str, cache: dict) -> None:
    import json

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write-then-rename so a concurrent reader never sees a partial file.
        # Two ranks racing here both produce a valid cache; one simply wins.
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(cache, fh)
        os.replace(tmp, path)
    except OSError:
        pass          # a cache we cannot write is not a reason to fail a run


def count_file_rows(path: str, chunk_size: int = 8 << 20) -> int:
    """Lines in one file. Chunked binary reads — ~1.3 GB/s here."""
    total = 0
    with open(path, "rb", buffering=0) as fh:
        while chunk := fh.read(chunk_size):
            total += chunk.count(b"\n")
    return total


def count_rows(
    files: Sequence[str], *, workers: int = 8, use_cache: bool = True
) -> List[int]:
    """Row count per file, in the order given. Cached across runs."""
    cache_path = _rowcount_cache_path()
    cache = _load_rowcount_cache(cache_path) if use_cache else {}
    counts: List[Optional[int]] = [None] * len(files)
    todo: List[Tuple[int, str, str]] = []

    for i, f in enumerate(files):
        key = os.path.abspath(f)
        try:
            st = os.stat(f)
        except OSError as exc:
            raise SystemExit(f"Cannot stat {f}: {exc}") from None
        entry = cache.get(key)
        # Size and mtime together are enough: a rewritten file changes at least
        # one of them, and a stale count would silently mis-scale the epoch.
        if (use_cache and isinstance(entry, list) and len(entry) == 3
                and entry[0] == st.st_size and entry[1] == st.st_mtime_ns):
            counts[i] = int(entry[2])
        else:
            todo.append((i, f, key))

    if todo:
        from concurrent.futures import ThreadPoolExecutor

        # I/O bound, so threads are enough and the GIL is not in the way.
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for (i, f, key), n in zip(todo, pool.map(
                    lambda t: count_file_rows(t[1]), todo)):
                counts[i] = n
                st = os.stat(f)
                cache[key] = [st.st_size, st.st_mtime_ns, n]
        if use_cache:
            _save_rowcount_cache(cache_path, cache)

    return [int(c) for c in counts]


def epoch_examples(
    files: Sequence[str],
    *,
    max_rows_per_file: Optional[int] = None,
    row_keep_prob: float = 1.0,
    workers: int = 8,
    use_cache: bool = True,
) -> int:
    """Examples one pass over ``files`` yields, as the loader would see them.

    Mirrors `StreamingTripleDataset`: `max_rows_per_file` truncates each file
    independently, and `row_keep_prob` drops rows at random (so this is an
    expectation, not an exact count, when it is below 1).
    """
    counts = count_rows(files, workers=workers, use_cache=use_cache)
    if max_rows_per_file is not None:
        counts = [min(c, int(max_rows_per_file)) for c in counts]
    total = sum(counts)
    if row_keep_prob < 1.0:
        total = int(total * float(row_keep_prob))
    return total
