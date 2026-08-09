"""`kgfm viz` — project a checkpoint's learned h / t embeddings to 2D.

The scorer is `sum(h_d * r_d * t_d)` over embeddings of raw strings, so what
the model actually learned is the geometry of that embedding space. This
command samples entity strings from the data, encodes them with a trained
checkpoint, reduces to two dimensions, and writes the coordinates as JSON for
`kgfm report` to plot.

Two things make the plot readable rather than a blob:

- **Points are deduplicated by text.** The corpus repeats the same entity
  across many triples; plotting per-triple would just weight frequent entities
  more heavily without adding information.
- **Head and tail are kept apart.** `DistMultScorer` normalizes h and t but not
  r, and h and t play different roles in the score, so whether the two clouds
  separate is itself a result worth seeing.

Each point carries labels the model never trains on — its role (h/t), its RDF
node type, and the relation of the triple it came from — so any structure the
plot shows along those is learned from the text, not memorised from the label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .data import StreamingTripleDataset, read_file_list

REDUCERS = ("auto", "pca", "umap")
DEFAULT_MAX_POINTS = 4000


def _collect_entities(
    files: Sequence[str],
    max_points: int,
    *,
    seed: int = 0,
    max_text_len: int = 512,
) -> List[Tuple[str, str, str, str]]:
    """Sample unique ``(text, role, node type, relation)``, balanced h vs t.

    The relation is ``rel_text`` (the actual predicate) from the triple the
    entity was first seen in — *not* ``rel_type``, which is the RDF term kind
    and is ``URIRef`` for essentially every row. An entity can occur under
    several relations, so this is a label rather than a property, but with
    |R| ~ 20 it is by far the most informative thing to colour by here; the
    node-type column is almost always ``URIRef`` too.
    """
    per_role = max(1, max_points // 2)
    ds = StreamingTripleDataset(
        files=list(files), shuffle_files=True, shuffle_buffer=4096,
        row_keep_prob=1.0, max_text_len=max_text_len, seed=seed,
    )
    seen: Dict[str, set] = {"h": set(), "t": set()}
    out: List[Tuple[str, str, str, str]] = []
    for tri in ds:
        for role, text, ntype in (
            ("h", tri.n1_text, tri.n1_type),
            ("t", tri.n2_text, tri.n2_type),
        ):
            if len(seen[role]) >= per_role or text in seen[role]:
                continue
            seen[role].add(text)
            out.append((text, role, ntype or "?", tri.rel_text or "?"))
        if len(seen["h"]) >= per_role and len(seen["t"]) >= per_role:
            break
    return out


@torch.no_grad()
def _encode(scorer, texts: Sequence[str], device, batch_size: int = 256):
    """Encode in the same space the scorer ranks in (normalized when it is)."""
    chunks = []
    for i in range(0, len(texts), batch_size):
        emb = scorer.encode(list(texts[i : i + batch_size]))
        chunks.append(scorer._maybe_norm(emb, scorer.normalize).float().cpu())
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def _pca(x: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
    """2-D PCA via SVD. Returns coordinates and explained-variance ratios."""
    centered = x - x.mean(dim=0, keepdim=True)
    # full_matrices=False keeps this cheap for n >> d and d >> n alike.
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    var = (s ** 2)
    total = float(var.sum()) or 1.0
    return coords, [float(v) / total for v in var[:2]]


def _umap(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    import umap

    reducer = umap.UMAP(n_components=2, random_state=seed, init="random")
    return torch.from_numpy(reducer.fit_transform(x.numpy()))


def available_reducer(reducer: str = "auto") -> str:
    if reducer != "auto":
        return reducer
    try:
        import umap  # noqa: F401

        return "umap"
    except ImportError:
        return "pca"


def build_projection(
    ckpt_path: str,
    files: Sequence[str],
    *,
    reducer: str = "auto",
    max_points: int = DEFAULT_MAX_POINTS,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> dict:
    """Encode sampled entities from ``files`` and project them to 2D."""
    from .eval import _load_scorer_from_checkpoint
    from .utils import pick_free_gpu

    device = device or pick_free_gpu()
    scorer = _load_scorer_from_checkpoint(ckpt_path, device)

    sampled = _collect_entities(files, max_points, seed=seed)
    if not sampled:
        raise SystemExit("No entities sampled — are the data files readable?")
    texts = [t for t, _, _, _ in sampled]
    emb = _encode(scorer, texts, device)

    resolved = available_reducer(reducer)
    explained: Optional[List[float]] = None
    if resolved == "umap":
        try:
            coords = _umap(emb, seed=seed)
        except ImportError:
            resolved = "pca"
    if resolved == "pca":
        coords, explained = _pca(emb)

    ckpt_cfg = torch.load(ckpt_path, map_location="cpu").get("config", {}) or {}
    record = {
        "method": "kgfm",
        "encoder": ckpt_cfg.get("encoder"),
        "freeze_encoder": bool(ckpt_cfg.get("freeze_encoder")),
        "ckpt": str(ckpt_path),
        "reducer": resolved,
        "dim": int(emb.shape[1]),
        "n_points": len(sampled),
        "seed": seed,
        "points": {
            "x": [round(float(v), 5) for v in coords[:, 0]],
            "y": [round(float(v), 5) for v in coords[:, 1]],
            "role": [r for _, r, _, _ in sampled],
            "type": [ty for _, _, ty, _ in sampled],
            "relation": [rl for _, _, _, rl in sampled],
        },
    }
    if explained is not None:
        record["explained_variance_ratio"] = [round(v, 5) for v in explained]
    return record


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True,
                   help="Checkpoint whose embedding space to project.")
    p.add_argument("--test-list", default="list_chembl/test.txt",
                   help="TSV path list to sample entities from.")
    p.add_argument("--out", default=None,
                   help="Where to write the JSON (default: alongside the "
                        "checkpoint's run directory as embeddings_<tag>.json).")
    p.add_argument("--reducer", default="auto", choices=REDUCERS,
                   help="Dimensionality reduction. 'auto' uses UMAP when "
                        "installed, else PCA.")
    p.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS,
                   help=f"Total points to plot, split evenly between head and "
                        f"tail entities (default {DEFAULT_MAX_POINTS}). Kept "
                        f"in the low thousands so the report stays light.")
    p.add_argument("--seed", type=int, default=0)


def default_out_path(ckpt_path: str) -> Path:
    """`<run>/kgfm_ckpts_<tag>/best.pt` -> `<run>/embeddings_<tag>.json`."""
    ckpt = Path(ckpt_path)
    ckpt_dir = ckpt.parent
    tag = ckpt_dir.name.replace("kgfm_ckpts_", "") or "kgfm"
    return ckpt_dir.parent / f"embeddings_{tag}.json"


def run_from_args(args: argparse.Namespace) -> dict:
    files = read_file_list(args.test_list)
    if not files:
        raise SystemExit(f"No files listed in {args.test_list}")
    record = build_projection(
        args.ckpt, files,
        reducer=args.reducer, max_points=args.max_points, seed=args.seed,
    )
    out_path = Path(args.out) if args.out else default_out_path(args.ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    # Recorded only when this lands in a run directory, so `kgfm report` can
    # show how the projection was produced.
    if (out_path.parent / "meta.json").is_file():
        from .runs import RunLogger

        RunLogger(out_path.parent).record_command(
            tag=out_path.stem.replace("embeddings_", ""),
            note="embedding projection",
        )
    ev = record.get("explained_variance_ratio")
    print(f"[viz] {record['n_points']} points, reducer={record['reducer']}"
          + (f", explained variance={ev}" if ev else ""))
    print(f"[viz] wrote {out_path}")
    return record


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="kgfm viz",
                                description=__doc__.splitlines()[0])
    add_arguments(p)
    run_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
