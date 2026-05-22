"""Convert kgfm chembl TSVs into an entity-ID KG that ULTRA / MOTIF can consume.

Input
-----
The three list files at ``list_chembl/{train,valid,test}.txt`` (one TSV path per
line). Triples are streamed via ``kgfm.data.StreamingTripleDataset`` so that
huge corpora are processed without loading them into memory.

Output (default: ``benchmarks/chembl_kg/``)
-------------------------------------------
- ``train.txt`` ``valid.txt`` ``test.txt`` — tab-separated ``head\\trel\\ttail``
  with raw entity / relation strings (this is what most ULTRA / MOTIF data
  loaders expect; they map strings to integer IDs internally).
- ``entities.dict`` — ``id\\tname`` for every distinct entity string seen.
- ``relations.dict`` — same for relations.
- ``stats.json`` — counts and sampling parameters used.

Triples are *capped* per split (``--max-train``, ``--max-valid``, ``--max-test``)
because chembl in full would blow up the entity vocabulary far beyond what
ULTRA can hold in memory at inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Allow running this script directly without `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kgfm.data import StreamingTripleDataset, read_file_list  # noqa: E402


def _normalize(s: str) -> str:
    """Collapse whitespace so the same entity string gets a single ID."""
    return " ".join(s.split())[:512]


def _stream_split(
    list_path: str,
    out_path: str,
    *,
    max_rows: Optional[int],
    seed: int,
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    freeze_vocab: bool,
) -> int:
    """Stream one split, write ``head\\trel\\ttail``, update / lookup vocab.

    When ``freeze_vocab`` is True, triples that reference an unseen entity
    or relation are skipped (the conventional "filtered transductive" setup
    for valid/test).
    """
    files = read_file_list(list_path)
    # File shuffling spreads the row cap across the whole list — without it,
    # `--max-train N` would drain the first file or two and miss most of the
    # corpus's entity / relation diversity.
    ds = StreamingTripleDataset(
        files=files,
        shuffle_files=True,
        shuffle_buffer=4096,
        row_keep_prob=1.0,
        seed=seed,
    )

    n_kept = 0
    n_skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for tri in ds:
            h = _normalize(tri.n1_text)
            r = _normalize(tri.rel_text)
            t = _normalize(tri.n2_text)
            if not h or not r or not t:
                continue

            if freeze_vocab:
                if h not in ent2id or t not in ent2id or r not in rel2id:
                    n_skipped += 1
                    continue
            else:
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)

            out.write(f"{h}\t{r}\t{t}\n")
            n_kept += 1
            if max_rows is not None and n_kept >= max_rows:
                break

    return n_kept


def _dump_dict(path: str, name2id: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for name, idx in sorted(name2id.items(), key=lambda kv: kv[1]):
            f.write(f"{idx}\t{name}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-list", default="list_chembl/train.txt")
    p.add_argument("--valid-list", default="list_chembl/valid.txt")
    p.add_argument("--test-list", default="list_chembl/test.txt")
    p.add_argument("--out-dir", default="benchmarks/chembl_kg")
    p.add_argument("--max-train", type=int, default=2_000_000,
                   help="Cap training triples (vocabulary is built on these).")
    p.add_argument("--max-valid", type=int, default=20_000)
    p.add_argument("--max-test", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--strict-transductive", action="store_true",
        help="Drop valid/test triples that reference entities or relations "
             "not seen in train. Default is inductive: valid/test may "
             "introduce new entities (chembl shards have near-disjoint "
             "entity sets, so the strict setting yields almost no held-out "
             "triples on this corpus).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    print(f"[prep] streaming train -> {args.out_dir}/train.txt "
          f"(max={args.max_train:,})")
    n_train = _stream_split(
        args.train_list, os.path.join(args.out_dir, "train.txt"),
        max_rows=args.max_train, seed=args.seed,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=False,
    )
    print(f"[prep] train: kept={n_train:,} "
          f"|E|={len(ent2id):,} |R|={len(rel2id):,}")

    freeze = args.strict_transductive
    mode = "transductive" if freeze else "inductive"
    print(f"[prep] streaming valid ({mode}: freeze_vocab={freeze})")
    n_valid = _stream_split(
        args.valid_list, os.path.join(args.out_dir, "valid.txt"),
        max_rows=args.max_valid, seed=args.seed + 1,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=freeze,
    )

    print(f"[prep] streaming test ({mode}: freeze_vocab={freeze})")
    n_test = _stream_split(
        args.test_list, os.path.join(args.out_dir, "test.txt"),
        max_rows=args.max_test, seed=args.seed + 2,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=freeze,
    )

    _dump_dict(os.path.join(args.out_dir, "entities.dict"), ent2id)
    _dump_dict(os.path.join(args.out_dir, "relations.dict"), rel2id)

    stats = {
        "n_entities": len(ent2id),
        "n_relations": len(rel2id),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "max_train": args.max_train,
        "max_valid": args.max_valid,
        "max_test": args.max_test,
        "seed": args.seed,
        "mode": mode,
        "train_list": args.train_list,
        "valid_list": args.valid_list,
        "test_list": args.test_list,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[prep] done. {stats}")


if __name__ == "__main__":
    main()
