#!/usr/bin/env python
"""Generate the ``list_*/`` file lists that `kgfm train` / `kgfm bench` consume.

A list directory is three plain text files — ``train.txt`` / ``valid.txt`` /
``test.txt`` — holding one TSV path per line (`#` comments and blank lines are
skipped by `kgfm.data.read_file_list`). Splits in kgfm are **file-level**, never
row-level, so choosing which files go where *is* the split.

WHY THIS SCRIPT EXISTS AT ALL
`kgfm train` can split on its own: given no lists it hash-buckets whatever
`--data-root`/`--pattern` discovers. But every benchmark config names explicit
lists instead (`list_chembl/train.txt`, ...), because a run has to be
reproducible against a *frozen* set of files: `data/` is a live mirror, so a
file appearing or disappearing would silently move the boundary between train
and test and make two runs incomparable. This script is how those frozen lists
are produced, and `--check` is how you confirm a checked-in list still matches
the rule that made it.

    # Regenerate list_chembl (the default source), byte-identical to the
    # version in the repo
    python preprocess_list/make_lists.py --source chembl --out-dir list_chembl

    # Any other data/<name> subtree, same rule
    python preprocess_list/make_lists.py --source uniprot --out-dir list_uniprot
    python preprocess_list/make_lists.py --source chebi,rhea --out-dir list_chem
    python preprocess_list/make_lists.py --source all --out-dir list_everything

    # Look before writing, and verify an existing list still matches the rule
    python preprocess_list/make_lists.py --source chembl --dry-run --count-rows
    python preprocess_list/make_lists.py --source chembl --check list_chembl

THE TWO SPLIT MODES ARE NOT INTERCHANGEABLE
`hash` (default) is the same function the trainer's own fallback uses
(`kgfm.data.split_files_three_way`), so a list built here and a list the trainer
derives on the fly agree exactly. It hashes the **path string**, which is why
this script writes repo-relative paths (`data/chembl/latest/x.tsv`) and refuses
an absolute `--data-root` in hash mode: `/data1/.../data/chembl/latest/x.tsv`
hashes to a different bucket and would produce a different, silently
incompatible split.

`sequential` takes files in sorted order — the first N to train, the next N to
valid, the rest/next N to test. Use it for small smoke sets where you want a
specific handful of files, not a random-looking spread. It is how
`list_large/` was made (the first 60 files of amrportal+bacdive+biomodels,
40/10/10).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import random
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Import from the package itself rather than reimplementing the split: the
# whole point of `hash` mode is that it agrees with the trainer bit for bit.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgfm.data import (  # noqa: E402
    _iter_tsv_rows,
    discover_tsv_files,
    read_file_list,
    split_files_three_way,
)
from kgfm.utils import file_split_bucket  # noqa: E402

SPLITS = ("train", "valid", "test")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def discover(
    data_root: str, sources: Sequence[str], pattern: str
) -> List[str]:
    """All TSVs under the named `data/<source>` subtrees, sorted globally.

    Sorting across sources rather than per source matters for `sequential`
    mode, where the order *is* the split. `all` means the whole data root.
    """
    if len(sources) == 1 and sources[0] == "all":
        return discover_tsv_files(data_root, pattern)
    seen: Dict[str, None] = {}
    for src in sources:
        root = os.path.join(data_root, src)
        if not os.path.isdir(root):
            raise SystemExit(
                f"No such data source: {root}\n"
                f"Available: {', '.join(list_sources(data_root)) or '(none)'}"
            )
        found = discover_tsv_files(root, pattern)
        if not found:
            raise SystemExit(
                f"{root} exists but matched no files with pattern "
                f"{pattern!r}. Check --pattern; nested sources such as "
                f"bioportal/<ontology>/latest are covered by the default "
                f"'**/latest/*.tsv'."
            )
        for f in found:
            seen[f] = None
    return sorted(seen)


def list_sources(data_root: str) -> List[str]:
    try:
        return sorted(
            d.name for d in os.scandir(data_root) if d.is_dir()
        )
    except OSError:
        return []


def apply_filters(
    files: Sequence[str],
    *,
    exclude: Sequence[str],
    include: Sequence[str],
    min_bytes: int,
) -> Tuple[List[str], Dict[str, int]]:
    """Drop files by glob and by size. Returns (kept, per-reason drop counts)."""
    dropped = {"exclude": 0, "include": 0, "min_bytes": 0, "missing": 0}
    kept: List[str] = []
    for f in files:
        base = os.path.basename(f)
        if include and not any(
            fnmatch.fnmatch(f, p) or fnmatch.fnmatch(base, p) for p in include
        ):
            dropped["include"] += 1
            continue
        if any(
            fnmatch.fnmatch(f, p) or fnmatch.fnmatch(base, p) for p in exclude
        ):
            dropped["exclude"] += 1
            continue
        try:
            size = os.path.getsize(f)
        except OSError:
            dropped["missing"] += 1
            continue
        if size < min_bytes:
            dropped["min_bytes"] += 1
            continue
        kept.append(f)
    return kept, dropped


def parse_size(text: str) -> int:
    """Parse `150GiB` / `105G` / `2TB` / a raw byte count into bytes."""
    s = str(text).strip().replace("_", "")
    units = (("KIB", 1 << 10), ("MIB", 1 << 20), ("GIB", 1 << 30),
             ("TIB", 1 << 40), ("KB", 10 ** 3), ("MB", 10 ** 6),
             ("GB", 10 ** 9), ("TB", 10 ** 12),
             ("K", 1 << 10), ("M", 1 << 20), ("G", 1 << 30), ("T", 1 << 40),
             ("B", 1))
    up = s.upper()
    for suffix, mult in units:
        if up.endswith(suffix):
            head = up[:-len(suffix)].strip()
            if not head:
                continue
            try:
                return int(float(head) * mult)
            except ValueError:
                break
    try:
        return int(float(s))
    except ValueError:
        raise SystemExit(
            f"Cannot parse a size from {text!r}. Use forms like 150GiB, "
            f"105G, 2TB, or a plain byte count."
        ) from None


def sample_to_size(
    files: Sequence[str], target_bytes: int, seed: int,
    probe_rows: int = 0,
) -> Tuple[List[str], int, int]:
    """Draw files uniformly at random (no replacement) until they total
    ``target_bytes``, then stop. Returns (sampled, achieved bytes).

    Uniform **over files**, not over bytes: every file in the pool is equally
    likely, which is what makes the result a random sample of the corpus
    rather than a size-biased one. Since file sizes here span 0 B to 2.1 GiB
    the achieved total overshoots the target by up to one file, and the file
    *count* is whatever it takes — that variability is the sampling, not a
    defect.

    The draw is over the pool as given, so `--source all` makes it a random
    sample across every data source and the result is a heterogeneous,
    multi-domain corpus. Combined with a split it is still the split that
    decides train/valid/test; this only decides which files are in play.
    """
    pool = list(files)
    random.Random(seed).shuffle(pool)
    taken: List[str] = []
    total = 0
    rejected = 0
    for f in pool:
        if total >= target_bytes:
            break
        try:
            size = os.path.getsize(f)
        except OSError:
            continue
        # Validate here rather than over the whole pool: the draw needs ~200
        # files out of ~20,000, so checking during the draw is two orders of
        # magnitude less I/O than checking everything first, and the target is
        # still met with files that actually parse.
        if probe_rows and not list(islice(_iter_tsv_rows(f), probe_rows)):
            rejected += 1
            continue
        total += size
        taken.append(f)
    if total < target_bytes:
        raise SystemExit(
            f"The pool holds only {_human(total)} of usable files but "
            f"--sample-bytes asked for {_human(target_bytes)}. Widen "
            f"--source, or lower the target."
        )
    # Sorted so the written list is stable and diffable; the *selection* was
    # random, the order within it carries no information.
    return sorted(taken), total, rejected


def validate(files: Sequence[str], probe_rows: int) -> Tuple[List[str], List[str]]:
    """Keep only files whose first rows satisfy the 6-column data contract.

    `_iter_tsv_rows` is the loader's own parser, so "yields nothing" here means
    "yields nothing during training" — a file that would occupy a slot in the
    split and contribute no examples. Cheap because it stops after
    `probe_rows`; a file is judged on its head, not read whole.
    """
    ok: List[str] = []
    bad: List[str] = []
    for f in files:
        rows = list(islice(_iter_tsv_rows(f), max(1, probe_rows)))
        (ok if rows else bad).append(f)
    return ok, bad


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def split_hash(
    files: Sequence[str], n_buckets: int, valid_buckets: int, test_buckets: int
) -> Dict[str, List[str]]:
    train, valid, test = split_files_three_way(
        files,
        valid_buckets=valid_buckets,
        test_buckets=test_buckets,
        n_buckets=n_buckets,
    )
    return {"train": train, "valid": valid, "test": test}


def split_sequential(
    files: Sequence[str],
    n_train: Optional[int],
    n_valid: Optional[int],
    n_test: Optional[int],
    ratios: Optional[Tuple[float, float, float]],
) -> Dict[str, List[str]]:
    """Take files in the order given: first n_train, then n_valid, then n_test.

    With `--ratios` the counts are derived from the pool size instead; train
    absorbs the rounding so the three counts always sum to len(files).
    """
    total = len(files)
    if ratios is not None:
        rv, rt = ratios[1], ratios[2]
        n_valid = int(round(total * rv))
        n_test = int(round(total * rt))
        n_train = total - n_valid - n_test
    else:
        n_valid = total if n_valid is None else n_valid
        n_test = total if n_test is None else n_test
        n_train = total if n_train is None else n_train
    if n_train + n_valid + n_test > total:
        raise SystemExit(
            f"Asked for {n_train}+{n_valid}+{n_test} = "
            f"{n_train + n_valid + n_test} files but only {total} are "
            f"available after filtering. Lower the counts or widen --source."
        )
    a, b = n_train, n_train + n_valid
    return {
        "train": list(files[:a]),
        "valid": list(files[a:b]),
        "test": list(files[b:b + n_test]),
    }


def split_random(
    files: Sequence[str],
    seed: int,
    n_train: Optional[int],
    n_valid: Optional[int],
    n_test: Optional[int],
    ratios: Optional[Tuple[float, float, float]],
) -> Dict[str, List[str]]:
    """Shuffle with an explicit seed, then split sequentially.

    Deterministic given (file set, seed) — but *not* given the seed alone: add
    or remove one file upstream and every assignment can move. `hash` mode is
    stable under that, which is why it is the default.
    """
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    out = split_sequential(shuffled, n_train, n_valid, n_test, ratios)
    return {k: sorted(v) for k, v in out.items()}


def split_per_source(
    pools: Dict[str, List[str]],
    counts: Dict[str, Optional[int]],
) -> Dict[str, List[str]]:
    """One source list per split — "train on X, evaluate on Y".

    Splits are filled in train -> valid -> test order and a file already
    claimed by an earlier split is skipped, so sharing a source between two
    splits partitions it instead of leaking it. With disjoint sources the
    exclusion never fires and each split simply takes its own files.
    """
    claimed: set = set()
    out: Dict[str, List[str]] = {}
    for name in SPLITS:
        pool = pools.get(name, [])
        avail = [f for f in pool if f not in claimed]
        want = counts.get(name)
        take = avail if want is None else avail[:want]
        if want is not None and len(take) < want:
            raise SystemExit(
                f"--{name}-files asked for {want} files but only "
                f"{len(take)} were available for {name} after removing the "
                f"{len(claimed)} already claimed by an earlier split. Give "
                f"{name} its own --{name}-source, or lower the count."
            )
        if want is None and pool and not avail:
            # The give-away symptom of sharing a source without saying how to
            # divide it: an earlier split, also uncapped, swallowed the pool.
            raise SystemExit(
                f"{name} would be empty: every file in its source was already "
                f"claimed by an earlier split. When two splits share a "
                f"source you have to say how to divide it — pass "
                f"--train-files/--valid-files/--test-files, or give {name} a "
                f"different --{name}-source."
            )
        claimed.update(take)
        out[name] = list(take)
    return out


def assert_disjoint(splits: Dict[str, List[str]]) -> None:
    """A file in two splits is train/test contamination — refuse to write it.

    File-level splitting is the only isolation kgfm has: every row of a train
    file is a training row, so one shared file leaks a whole entity population
    into the evaluation set.
    """
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            both = sorted(set(splits[a]) & set(splits[b]))
            if both:
                shown = ", ".join(both[:5])
                more = "" if len(both) <= 5 else f" (+{len(both) - 5} more)"
                raise SystemExit(
                    f"{len(both)} file(s) landed in both {a} and {b}: "
                    f"{shown}{more}\nSplits must be disjoint — kgfm splits at "
                    f"the file level, so a shared file puts every one of its "
                    f"rows on both sides."
                )


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def header_lines(argv: Sequence[str], n_files: int, split_desc: str) -> List[str]:
    """Provenance comments. `read_file_list` skips them, so they are free."""
    return [
        "# Generated by preprocess_list/make_lists.py — do not edit by hand.",
        f"# command: python {' '.join(argv)}",
        f"# pool: {n_files} files | split: {split_desc}",
    ]


def write_lists(
    out_dir: str,
    splits: Dict[str, List[str]],
    header: Optional[List[str]],
    force: bool,
) -> None:
    out = Path(out_dir)
    existing = [out / f"{s}.txt" for s in SPLITS if (out / f"{s}.txt").exists()]
    if existing and not force:
        raise SystemExit(
            f"{out_dir} already holds "
            f"{', '.join(p.name for p in existing)}. Pass --force to "
            f"overwrite, or --check {out_dir} to compare without writing."
        )
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        body = list(splits[name])
        lines = (header or []) + body
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_against(out_dir: str, splits: Dict[str, List[str]]) -> int:
    """Compare the generated split with a checked-in list dir. Returns exit code.

    Compares as sets *and* as ordered lists, and ignores comment lines, because
    `read_file_list` ignores them too — a header-only difference is not a real
    difference.
    """
    rc = 0
    for name in SPLITS:
        path = Path(out_dir) / f"{name}.txt"
        if not path.exists():
            print(f"  {name}: MISSING {path}")
            rc = 1
            continue
        have = read_file_list(str(path))
        want = splits[name]
        if have == want:
            print(f"  {name}: identical ({len(want)} files)")
            continue
        rc = 1
        sh, sw = set(have), set(want)
        if sh == sw:
            print(f"  {name}: same {len(want)} files, DIFFERENT ORDER")
            continue
        print(f"  {name}: DIFFERS (on disk {len(have)}, generated {len(want)})")
        for label, diff in (("only on disk", sh - sw), ("only generated", sw - sh)):
            if diff:
                shown = sorted(diff)
                tail = "" if len(shown) <= 5 else f" (+{len(shown) - 5} more)"
                print(f"    {label}: {', '.join(shown[:5])}{tail}")
    return rc


def show_sources(splits: Dict[str, List[str]]) -> None:
    """Per-split file counts and bytes by `data/<source>`.

    Only interesting for a multi-source list, and then it is the first thing
    to check: splitting at the file level over a heterogeneous pool can put a
    source almost entirely into valid or test, which makes the evaluation a
    different domain from the training data rather than a held-out sample of
    it. That may be what you want (kgfm is inductive by design) but it should
    be a decision, not a surprise.
    """
    per: Dict[str, Dict[str, int]] = {}
    for name in SPLITS:
        for f in splits[name]:
            parts = f.split(os.sep)
            src = parts[1] if len(parts) > 1 else "?"
            row = per.setdefault(src, {s: 0 for s in SPLITS})
            row[name] += 1
    if not per:
        return
    byte_tot: Dict[str, int] = {}
    for name in SPLITS:
        for f in splits[name]:
            parts = f.split(os.sep)
            src = parts[1] if len(parts) > 1 else "?"
            try:
                byte_tot[src] = byte_tot.get(src, 0) + os.path.getsize(f)
            except OSError:
                pass
    print(f"\n{'source':<22}{'train':>7}{'valid':>7}{'test':>7}{'bytes':>12}")
    order = sorted(per, key=lambda s: -byte_tot.get(s, 0))
    for src in order:
        r = per[src]
        print(f"{src:<22}{r['train']:>7}{r['valid']:>7}{r['test']:>7}"
              f"{_human(byte_tot.get(src, 0)):>12}")
    print(f"{'sources':<22}{len(order):>7}")
    # The failure mode worth naming out loud.
    for name in ("valid", "test"):
        srcs = {f.split(os.sep)[1] for f in splits[name] if os.sep in f}
        if len(srcs) == 1 and len(order) > 1:
            print(f"WARNING: {name} is entirely from one source "
                  f"({next(iter(srcs))}) while the list spans {len(order)}. "
                  f"That measures transfer to that source, not held-out "
                  f"performance on the corpus.")


def report(
    splits: Dict[str, List[str]],
    *,
    count_rows: bool,
    row_workers: int,
) -> None:
    total_files = sum(len(v) for v in splits.values())
    print(f"\n{'split':<8}{'files':>8}{'bytes':>14}" + (f"{'rows':>16}" if count_rows else ""))
    counts: Dict[str, int] = {}
    if count_rows:
        from kgfm.data import count_rows as _count_rows
        flat = [f for name in SPLITS for f in splits[name]]
        per_file = dict(zip(flat, _count_rows(flat, workers=row_workers)))
    for name in SPLITS:
        files = splits[name]
        nbytes = sum(os.path.getsize(f) for f in files)
        line = f"{name:<8}{len(files):>8}{_human(nbytes):>14}"
        if count_rows:
            n = sum(per_file[f] for f in files)
            counts[name] = n
            line += f"{n:>16,}"
        print(line)
    print(f"{'total':<8}{total_files:>8}" + " " * 14
          + (f"{sum(counts.values()):>16,}" if count_rows else ""))
    if not any(splits[s] for s in ("valid", "test")):
        print("\nWARNING: valid and test are both empty — the trainer would "
              "have nothing to validate on. Widen --source or lower the "
              "bucket/count settings.")


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return str(n)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="make_lists.py",
        description="Build a list_*/ directory (train/valid/test.txt) from "
                    "the TSVs under data/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", default="chembl",
        help="Comma-separated data/<name> subtrees, or 'all' for the whole "
             "data root (default: chembl, i.e. what list_chembl/ is built "
             "from). Use --list-sources to see what is available.")
    p.add_argument(
        "--data-root", default="data",
        help="Root of the corpus mirror (default: data). Keep this relative: "
             "the hash split hashes the path string, so an absolute root "
             "produces a different split.")
    p.add_argument(
        "--pattern", default="**/latest/*.tsv",
        help="Glob under <data-root>/<source>, recursive (default: "
             "'**/latest/*.tsv'; covers both data/chembl/latest and nested "
             "layouts such as data/bioportal/<ontology>/latest).")
    p.add_argument("--out-dir", default=None,
                   help="Where to write train/valid/test.txt. Omit with "
                        "--dry-run or --check.")
    p.add_argument("--train-source", default=None,
                   help="Give one split its own sources — 'train on X, "
                        "evaluate on Y'. Setting any of --train/valid/"
                        "test-source switches off --split entirely: each "
                        "split takes --<name>-files files from its own sorted "
                        "pool (all of them by default), filled in "
                        "train->valid->test order and skipping anything an "
                        "earlier split already took, so sharing a source "
                        "partitions it instead of leaking it.")
    p.add_argument("--valid-source", default=None,
                   help="See --train-source. Defaults to --source.")
    p.add_argument("--test-source", default=None,
                   help="See --train-source. Defaults to --source.")

    g = p.add_argument_group("split")
    g.add_argument("--split", default="hash",
                   choices=["hash", "sequential", "random"],
                   help="hash (default) = the trainer's own hash-bucket rule, "
                        "stable when files are added or removed; sequential = "
                        "first N / next N / next N in sorted order; random = "
                        "seeded shuffle then sequential.")
    g.add_argument("--n-buckets", type=int, default=10,
                   help="hash mode: number of buckets (default 10).")
    g.add_argument("--valid-buckets", type=int, default=1,
                   help="hash mode: buckets assigned to valid (default 1).")
    g.add_argument("--test-buckets", type=int, default=1,
                   help="hash mode: buckets assigned to test (default 1). "
                        "The default 1/1/10 is the 80/10/10 split.")
    g.add_argument("--train-files", type=int, default=None,
                   help="sequential/random: how many files to put in train.")
    g.add_argument("--valid-files", type=int, default=None,
                   help="sequential/random: how many files to put in valid.")
    g.add_argument("--test-files", type=int, default=None,
                   help="sequential/random: how many files to put in test.")
    g.add_argument("--ratios", default=None, metavar="TRAIN,VALID,TEST",
                   help="sequential/random: fractions instead of counts "
                        "(e.g. 0.8,0.1,0.1). Overrides --*-files.")
    g.add_argument("--random-seed", type=int, default=0,
                   help="random mode seed (default 0).")

    f = p.add_argument_group("file selection")
    f.add_argument("--max-files", type=int, default=None,
                   help="Cap the pool to the first N files (after sorting and "
                        "filtering, before splitting).")
    f.add_argument("--sample-bytes", default=None, metavar="SIZE",
                   help="Draw files uniformly at random (seeded by "
                        "--random-seed) until they total SIZE, e.g. 150GiB / "
                        "105G / 2TB. Use with --source all to build a random "
                        "multi-source corpus of a chosen scale. Applied after "
                        "the other filters and before the split; --max-files "
                        "is ignored when this is given.")
    f.add_argument("--exclude", action="append", default=[], metavar="GLOB",
                   help="Drop files matching this glob (matched against both "
                        "the full path and the basename). Repeatable.")
    f.add_argument("--include", action="append", default=[], metavar="GLOB",
                   help="Keep only files matching one of these globs. "
                        "Repeatable.")
    f.add_argument("--min-bytes", type=int, default=0,
                   help="Drop files smaller than this (default 0 = keep all).")
    f.add_argument("--validate", action="store_true",
                   help="Parse the head of every file with the loader's own "
                        "reader and drop files that yield no valid 6-column "
                        "row. Costs one small read per file.")
    f.add_argument("--probe-rows", type=int, default=5,
                   help="--validate: rows to read per file (default 5).")

    o = p.add_argument_group("output / inspection")
    o.add_argument("--dry-run", action="store_true",
                   help="Print the split and write nothing.")
    o.add_argument("--check", metavar="DIR", default=None,
                   help="Compare the generated split against an existing list "
                        "directory and exit non-zero if they differ. Writes "
                        "nothing; comment lines are ignored.")
    o.add_argument("--force", action="store_true",
                   help="Overwrite an existing --out-dir.")
    o.add_argument("--no-header", action="store_true",
                   help="Omit the '# Generated by ...' provenance comments. "
                        "Needed only to byte-match a hand-written list.")
    o.add_argument("--count-rows", action="store_true",
                   help="Also report rows per split. Uses the cached row "
                        "counter (~/.cache/kgfm/rowcounts.json), so it is "
                        "instant on a corpus already counted and ~1.3 GB/s "
                        "otherwise.")
    o.add_argument("--row-workers", type=int, default=8,
                   help="--count-rows: reader threads (default 8).")
    o.add_argument("--print-files", action="store_true",
                   help="Print every path in every split.")
    o.add_argument("--show-sources", action="store_true",
                   help="Break each split down by data source. Worth reading "
                        "for a random multi-source list: file-level splitting "
                        "means valid/test can end up dominated by sources "
                        "that are barely in train.")
    o.add_argument("--list-sources", action="store_true",
                   help="List the data/<name> subtrees and exit.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.list_sources:
        names = list_sources(args.data_root)
        if not names:
            raise SystemExit(f"No subdirectories under {args.data_root!r}.")
        print(f"{len(names)} sources under {args.data_root}/:")
        for n in names:
            print(f"  {n}")
        return 0

    if not args.out_dir and not (args.dry_run or args.check):
        raise SystemExit("Nothing to do: pass --out-dir, --dry-run or --check.")

    # The hash split hashes the path string, so the form of the path decides
    # the buckets. An absolute root would produce a split that no checked-in
    # list matches and that the trainer's own fallback would not reproduce.
    if args.split == "hash" and os.path.isabs(args.data_root):
        raise SystemExit(
            "--data-root must be relative in hash mode: the split hashes the "
            "path string, so an absolute root gives different buckets than "
            "every checked-in list and than the trainer's own fallback split. "
            "Run from the repo root with --data-root data, or use "
            "--split sequential."
        )

    sources = [s.strip() for s in args.source.split(",") if s.strip()]
    if not sources:
        raise SystemExit("--source is empty.")

    def prepare(srcs: Sequence[str], label: str) -> List[str]:
        found = discover(args.data_root, srcs, args.pattern)
        print(f"{label}: discovered {len(found)} files under "
              f"{args.data_root}/{{{','.join(srcs)}}}/{args.pattern}")
        found, dropped = apply_filters(
            found, exclude=args.exclude, include=args.include,
            min_bytes=args.min_bytes,
        )
        for reason, n in dropped.items():
            if n:
                print(f"  dropped {n} by {reason}")
        if args.sample_bytes is not None:
            # Sampling subsumes --validate here: the sampler checks each file
            # as it draws it, so only the drawn files are read.
            target = parse_size(args.sample_bytes)
            found, got, rejected = sample_to_size(
                found, target, args.random_seed,
                probe_rows=args.probe_rows if args.validate else 0,
            )
            print(f"  randomly sampled {len(found)} files totalling "
                  f"{_human(got)} (target {_human(target)}, "
                  f"seed {args.random_seed})")
            if rejected:
                print(f"  skipped {rejected} drawn file(s) with no valid "
                      f"6-column row")
        else:
            if args.validate:
                found, bad = validate(found, args.probe_rows)
                if bad:
                    print(f"  dropped {len(bad)} with no valid 6-column row: "
                          f"{', '.join(os.path.basename(b) for b in bad[:5])}"
                          + (f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""))
        if args.sample_bytes is None and args.max_files is not None:
            found = found[:args.max_files]
            print(f"  capped to first {len(found)} by --max-files")
        if not found:
            raise SystemExit(
                f"{label}: no files left after filtering; nothing to split.")
        return found

    per_split_sources = {
        "train": args.train_source,
        "valid": args.valid_source,
        "test": args.test_source,
    }
    if any(v is not None for v in per_split_sources.values()):
        if args.split != "hash":
            print(f"note: --split {args.split} is ignored — per-split "
                  f"--*-source assigns the files itself.")
        pools: Dict[str, List[str]] = {}
        for name in SPLITS:
            raw = per_split_sources[name] or args.source
            srcs = [s.strip() for s in raw.split(",") if s.strip()]
            pools[name] = prepare(srcs, name)
        splits = split_per_source(pools, {
            "train": args.train_files,
            "valid": args.valid_files,
            "test": args.test_files,
        })
        desc = ("per-source " + " ".join(
            f"{n}={per_split_sources[n] or args.source}"
            f"[{len(splits[n])}]" for n in SPLITS))
        files = sorted({f for v in splits.values() for f in v})
        assert_disjoint(splits)
        return finish(args, splits, files, desc)

    files = prepare(sources, "pool")

    ratios = None
    if args.ratios:
        parts = [float(x) for x in args.ratios.split(",")]
        if len(parts) != 3:
            raise SystemExit("--ratios needs three numbers, e.g. 0.8,0.1,0.1")
        ratios = (parts[0], parts[1], parts[2])

    if args.split == "hash":
        if args.n_buckets < args.valid_buckets + args.test_buckets:
            raise SystemExit(
                f"--n-buckets ({args.n_buckets}) must be at least "
                f"--valid-buckets + --test-buckets "
                f"({args.valid_buckets + args.test_buckets}); otherwise train "
                f"gets nothing.")
        splits = split_hash(files, args.n_buckets, args.valid_buckets,
                            args.test_buckets)
        desc = (f"hash n_buckets={args.n_buckets} "
                f"valid_buckets={args.valid_buckets} "
                f"test_buckets={args.test_buckets}")
    elif args.split == "sequential":
        splits = split_sequential(files, args.train_files, args.valid_files,
                                  args.test_files, ratios)
        desc = (f"sequential train/valid/test="
                f"{len(splits['train'])}/{len(splits['valid'])}/"
                f"{len(splits['test'])}")
    else:
        splits = split_random(files, args.random_seed, args.train_files,
                              args.valid_files, args.test_files, ratios)
        desc = (f"random seed={args.random_seed} train/valid/test="
                f"{len(splits['train'])}/{len(splits['valid'])}/"
                f"{len(splits['test'])}")

    assert_disjoint(splits)
    return finish(args, splits, files, desc)


def finish(
    args: argparse.Namespace,
    splits: Dict[str, List[str]],
    pool: Sequence[str],
    desc: str,
) -> int:
    """Report, then do exactly one of: check, dry-run, write."""
    report(splits, count_rows=args.count_rows, row_workers=args.row_workers)
    if args.show_sources:
        show_sources(splits)
    if args.print_files:
        for name in SPLITS:
            print(f"\n[{name}]")
            for f in splits[name]:
                print(f"  {f}")

    if args.check:
        print(f"\nchecking against {args.check}/ ({desc})")
        rc = check_against(args.check, splits)
        print("\nOK — the checked-in list matches this rule." if rc == 0
              else "\nMISMATCH — the checked-in list was not produced by this "
                   "rule (or the corpus changed since).")
        return rc

    if args.dry_run:
        print(f"\ndry run ({desc}); nothing written")
        return 0

    header = None if args.no_header else header_lines(
        [os.path.relpath(sys.argv[0], _REPO_ROOT)] + sys.argv[1:],
        len(pool), desc,
    )
    write_lists(args.out_dir, splits, header, args.force)
    print(f"\nwrote {args.out_dir}/{{train,valid,test}}.txt ({desc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
