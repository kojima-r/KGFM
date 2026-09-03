"""Convert kgfm ChEMBL TSVs into an entity-ID KG that ULTRA / MOTIF can load.

kgfm itself has no entity vocabulary — it encodes raw strings — so this step
exists purely to give the ID-based baselines something they can consume.

Output (default ``benchmarks/chembl_kg/``)
------------------------------------------
- ``train.txt`` / ``valid.txt`` / ``test.txt`` — ``head\\trel\\ttail`` with raw
  entity / relation strings (ULTRA and MOTIF map them to integer IDs).
- ``entities.dict`` / ``relations.dict`` — ``id\\tname``.
- ``stats.json`` — counts and the sampling parameters used.

Triples are capped per split because the full ChEMBL corpus would blow up the
entity vocabulary far beyond what ULTRA can hold at inference time.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from ..data import StreamingTripleDataset, read_file_list
from ..runs import RunLogger
from .config import BenchConfig


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
    """Stream one split, write ``head\\trel\\ttail``, update / look up vocab.

    With ``freeze_vocab`` set, triples referencing an unseen entity or relation
    are dropped — the conventional filtered-transductive setup for valid/test.
    """
    files = read_file_list(list_path)
    # File shuffling spreads the row cap across the whole list — without it,
    # `--max-train N` would drain the first file or two and miss most of the
    # corpus's entity / relation diversity.
    ds = StreamingTripleDataset(
        files=files, shuffle_files=True, shuffle_buffer=4096,
        row_keep_prob=1.0, seed=seed,
    )

    n_kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for tri in ds:
            h = _normalize(tri.n1_text)
            r = _normalize(tri.rel_text)
            t = _normalize(tri.n2_text)
            if not h or not r or not t:
                continue

            if freeze_vocab:
                if h not in ent2id or t not in ent2id or r not in rel2id:
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


def build_kg(cfg: BenchConfig) -> dict:
    """Build the KG in ``cfg.kg_dir`` and return its stats."""
    os.makedirs(cfg.kg_dir, exist_ok=True)

    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    print(f"[prep] streaming train -> {cfg.kg_dir}/train.txt (max={cfg.prep_max_train:,})")
    n_train = _stream_split(
        cfg.train_list, os.path.join(cfg.kg_dir, "train.txt"),
        max_rows=cfg.prep_max_train, seed=cfg.seed,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=False,
    )
    print(f"[prep] train: kept={n_train:,} |E|={len(ent2id):,} |R|={len(rel2id):,}")

    freeze = cfg.strict_transductive
    mode = "transductive" if freeze else "inductive"

    print(f"[prep] streaming valid ({mode}: freeze_vocab={freeze})")
    n_valid = _stream_split(
        cfg.valid_list, os.path.join(cfg.kg_dir, "valid.txt"),
        max_rows=cfg.prep_max_valid, seed=cfg.seed + 1,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=freeze,
    )

    print(f"[prep] streaming test ({mode}: freeze_vocab={freeze})")
    n_test = _stream_split(
        cfg.test_list, os.path.join(cfg.kg_dir, "test.txt"),
        max_rows=cfg.prep_max_test, seed=cfg.seed + 2,
        ent2id=ent2id, rel2id=rel2id, freeze_vocab=freeze,
    )

    _dump_dict(os.path.join(cfg.kg_dir, "entities.dict"), ent2id)
    _dump_dict(os.path.join(cfg.kg_dir, "relations.dict"), rel2id)

    stats = {
        "n_entities": len(ent2id),
        "n_relations": len(rel2id),
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "prep_max_train": cfg.prep_max_train,
        "prep_max_valid": cfg.prep_max_valid,
        "prep_max_test": cfg.prep_max_test,
        "seed": cfg.seed,
        "mode": mode,
        "train_list": cfg.train_list,
        "valid_list": cfg.valid_list,
        "test_list": cfg.test_list,
    }
    with open(os.path.join(cfg.kg_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[prep] done. {stats}")
    return stats


def run_step(cfg: BenchConfig, out_dir: Path, logger: RunLogger,
             *, reuse: bool = False) -> None:
    """Prep step: build the KG (or adopt an existing one) and record its stats.

    ``chembl_kg_stats.json`` in the run dir is both the record and the resume
    marker.
    """
    stats_copy = out_dir / "chembl_kg_stats.json"
    src_stats = Path(cfg.kg_dir) / "stats.json"

    def adopt() -> None:
        if src_stats.is_file():
            stats_copy.write_text(src_stats.read_text())

    if reuse:
        logger.log(f"reuse existing ChEMBL KG in {cfg.kg_dir} (no rebuild)")
        adopt()
        return
    if cfg.resume and stats_copy.is_file():
        logger.log("skip prep (resume: chembl_kg_stats.json exists)")
        return

    # The processed caches are pickles of the *previous* KG; leaving them
    # behind would make ULTRA / MOTIF silently evaluate stale data.
    for cache in ("processed", "processed_motif"):
        path = Path(cfg.kg_dir) / cache
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)

    with logger.step("prep"):
        build_kg(cfg)
    adopt()
