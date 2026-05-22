"""Tail-prediction evaluation: MRR, Hit@k, nDCG.

Two protocols are available, selected via ``protocol="pooled"|"filtered"``:

**pooled** (default — fast, memory-bounded)

1. Build a candidate pool of size ``pool_size`` by streaming a slice of the
   test files and collecting unique tail texts.
2. Pre-encode the candidate pool once.
3. For each evaluated test triple, score against the pool plus the true
   tail and rank the true tail.

**filtered** (standard KG eval)

1. Stream the *filter* files (typically train+valid+test) and materialize:
   - the full set of unique tail texts (the candidate vocabulary), and
   - a ``(h_text, r_text) -> {tail_text}`` index of all known true triples.
2. Pre-encode the full tail vocabulary.
3. For each evaluated test triple ``(h, r, t)``, score ``(h, r)`` against
   every tail, set the score of *other* known true tails for the same
   ``(h, r)`` to ``-inf``, then rank ``t`` among the remaining candidates.

Filtered ranking is the protocol used by ULTRA / MOTIF and most published
KG-completion numbers, so it is the right choice for cross-method
comparison. It is heavier than pooled because the vocabulary can be much
larger and because the filter index has to be materialized.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader

from .data import StreamingTripleDataset, collate_triples
from .model import DistMultScorer

PROTOCOLS = ("pooled", "filtered")


@torch.no_grad()
def _encode_tail_pool(
    scorer: DistMultScorer, texts: List[str], device: torch.device, batch_size: int = 256
) -> torch.Tensor:
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        emb = scorer.encode(chunk)
        emb = scorer._maybe_norm(emb, scorer.normalize)
        out.append(emb.to(device))
    if not out:
        return torch.empty(0, scorer.dim, device=device)
    return torch.cat(out, dim=0)


def build_candidate_pool(
    files: List[str],
    pool_size: int,
    max_text_len: int = 512,
    seed: int = 0,
) -> List[str]:
    """Stream a few test files and collect unique tail texts up to ``pool_size``."""
    ds = StreamingTripleDataset(
        files=files,
        shuffle_files=True,
        shuffle_buffer=1,
        row_keep_prob=1.0,
        max_text_len=max_text_len,
        seed=seed,
    )
    seen: Set[str] = set()
    pool: List[str] = []
    for tri in ds:
        if tri.n2_text in seen:
            continue
        seen.add(tri.n2_text)
        pool.append(tri.n2_text)
        if len(pool) >= pool_size:
            break
    return pool


def build_filter_index(
    files: Sequence[str],
    max_text_len: int = 512,
    seed: int = 0,
    max_tails: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[List[str], Dict[Tuple[str, str], Set[str]]]:
    """Stream files; return ``(tail_vocab, hr_to_tails)``.

    - ``tail_vocab`` — unique tail strings in insertion order.
    - ``hr_to_tails`` — every known ``(h_text, r_text) -> {tail_text}``.

    ``max_tails`` caps the candidate vocabulary size. Once reached the
    filter index keeps growing (so masking still works) but no new tails
    enter the candidate pool. Test-time tails outside the vocabulary are
    appended on the fly during evaluation.

    ``max_rows`` caps the total number of rows streamed. Use it on
    corpora where exhausting the input would be impractical (e.g.
    multi-TB ChEMBL streams). The candidate vocabulary will still be
    accurate w.r.t. the rows that were seen, just incomplete.
    """
    ds = StreamingTripleDataset(
        files=list(files),
        shuffle_files=True,
        shuffle_buffer=4096,
        row_keep_prob=1.0,
        max_text_len=max_text_len,
        seed=seed,
    )
    seen_tail: Set[str] = set()
    tail_vocab: List[str] = []
    hr_to_tails: Dict[Tuple[str, str], Set[str]] = {}
    n_rows = 0
    for tri in ds:
        if tri.n2_text not in seen_tail and (
            max_tails is None or len(tail_vocab) < max_tails
        ):
            seen_tail.add(tri.n2_text)
            tail_vocab.append(tri.n2_text)
        hr_to_tails.setdefault((tri.n1_text, tri.rel_text), set()).add(tri.n2_text)
        n_rows += 1
        if max_rows is not None and n_rows >= max_rows:
            break
    return tail_vocab, hr_to_tails


def _accumulate_metrics(
    ranks: torch.Tensor, k_hit: int, totals: Dict[str, float]
) -> None:
    ranks = ranks.to(torch.float32)
    totals["rr"] += (1.0 / ranks).sum().item()
    totals["hit"] += (ranks <= k_hit).sum().item()
    totals["ndcg"] += (1.0 / torch.log2(ranks + 1.0)).sum().item()


@torch.no_grad()
def _evaluate_pooled(
    scorer: DistMultScorer,
    test_files: List[str],
    device: torch.device,
    *,
    n_eval_triples: int,
    pool_size: int,
    batch_size: int,
    k_hit: int,
    max_text_len: int,
    num_workers: int,
    seed: int,
) -> dict:
    pool_texts = build_candidate_pool(
        test_files, pool_size=pool_size, max_text_len=max_text_len, seed=seed
    )
    if not pool_texts:
        return {"MRR": 0.0, f"Hit@{k_hit}": 0.0, "nDCG": 0.0, "n": 0,
                "protocol": "pooled"}

    pool_idx = {t: i for i, t in enumerate(pool_texts)}
    pool_emb = _encode_tail_pool(scorer, pool_texts, device)  # [P, D]

    eval_ds = StreamingTripleDataset(
        files=test_files,
        shuffle_files=True,
        shuffle_buffer=4096,
        row_keep_prob=1.0,
        max_text_len=max_text_len,
        seed=seed + 7,
    )
    loader = DataLoader(
        eval_ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_triples, persistent_workers=num_workers > 0,
    )

    totals = {"rr": 0.0, "hit": 0.0, "ndcg": 0.0}
    n = 0
    for batch in loader:
        h_text, r_text, t_text = batch["h_text"], batch["r_text"], batch["t_text"]
        h, r, t = scorer.encode_triple(h_text, r_text, t_text)
        h, r, t = h.to(device), r.to(device), t.to(device)

        hr = h * r  # [B, D]
        scores_pool = hr @ pool_emb.t()  # [B, P]
        true_score = (hr * t).sum(dim=-1, keepdim=True)  # [B, 1]
        B = len(t_text)
        # Don't double-count the true tail when it happens to be in pool.
        for i in range(B):
            j = pool_idx.get(t_text[i])
            if j is not None:
                scores_pool[i, j] = float("-inf")
        all_scores = torch.cat([scores_pool, true_score], dim=1)  # [B, P+1]
        gt = all_scores[:, -1:]  # last column is the true tail
        ranks = (all_scores > gt).sum(dim=1) + 1
        _accumulate_metrics(ranks, k_hit, totals)
        n += B
        if n >= n_eval_triples:
            break

    if n == 0:
        return {"MRR": 0.0, f"Hit@{k_hit}": 0.0, "nDCG": 0.0, "n": 0,
                "protocol": "pooled"}
    return {
        "MRR": totals["rr"] / n,
        f"Hit@{k_hit}": totals["hit"] / n,
        "nDCG": totals["ndcg"] / n,
        "n": n,
        "pool": len(pool_texts),
        "protocol": "pooled",
    }


@torch.no_grad()
def _evaluate_filtered(
    scorer: DistMultScorer,
    test_files: List[str],
    device: torch.device,
    *,
    filter_files: Sequence[str],
    n_eval_triples: int,
    batch_size: int,
    k_hit: int,
    max_text_len: int,
    num_workers: int,
    seed: int,
    max_tails: Optional[int],
    max_filter_rows: Optional[int],
) -> dict:
    tail_vocab, hr_to_tails = build_filter_index(
        filter_files, max_text_len=max_text_len, seed=seed,
        max_tails=max_tails, max_rows=max_filter_rows,
    )
    if not tail_vocab:
        return {"MRR": 0.0, f"Hit@{k_hit}": 0.0, "nDCG": 0.0, "n": 0,
                "protocol": "filtered"}

    tail_idx: Dict[str, int] = {t: i for i, t in enumerate(tail_vocab)}
    tail_emb = _encode_tail_pool(scorer, tail_vocab, device)  # [V, D]

    eval_ds = StreamingTripleDataset(
        files=test_files,
        shuffle_files=True,
        shuffle_buffer=4096,
        row_keep_prob=1.0,
        max_text_len=max_text_len,
        seed=seed + 7,
    )
    loader = DataLoader(
        eval_ds, batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_triples, persistent_workers=num_workers > 0,
    )

    totals = {"rr": 0.0, "hit": 0.0, "ndcg": 0.0}
    n = 0
    appended_unknown = 0
    for batch in loader:
        h_text, r_text, t_text = batch["h_text"], batch["r_text"], batch["t_text"]
        h, r, t = scorer.encode_triple(h_text, r_text, t_text)
        h, r = h.to(device), r.to(device)
        t = t.to(device)

        hr = h * r  # [B, D]
        scores = hr @ tail_emb.t()  # [B, V]
        true_score = (hr * t).sum(dim=-1, keepdim=True)  # [B, 1]
        B = len(t_text)

        # For each row, mask out other known true tails for (h,r), then
        # append the true tail score so we can rank it consistently
        # whether or not it was in the vocab.
        rows = []
        for i in range(B):
            row = scores[i].clone()
            others = hr_to_tails.get((h_text[i], r_text[i]), set()) - {t_text[i]}
            if others:
                idxs = [tail_idx[o] for o in others if o in tail_idx]
                if idxs:
                    row[torch.tensor(idxs, device=device, dtype=torch.long)] = float("-inf")
            # If the true tail IS in the vocab, mask its in-vocab score so
            # we don't double-count it after concatenating true_score.
            true_in_vocab = tail_idx.get(t_text[i])
            if true_in_vocab is not None:
                row[true_in_vocab] = float("-inf")
            else:
                appended_unknown += 1
            rows.append(row)
        masked = torch.stack(rows, dim=0)  # [B, V]
        all_scores = torch.cat([masked, true_score], dim=1)  # [B, V+1]

        gt = all_scores[:, -1:]
        ranks = (all_scores > gt).sum(dim=1) + 1
        _accumulate_metrics(ranks, k_hit, totals)
        n += B
        if n >= n_eval_triples:
            break

    if n == 0:
        return {"MRR": 0.0, f"Hit@{k_hit}": 0.0, "nDCG": 0.0, "n": 0,
                "protocol": "filtered"}
    return {
        "MRR": totals["rr"] / n,
        f"Hit@{k_hit}": totals["hit"] / n,
        "nDCG": totals["ndcg"] / n,
        "n": n,
        "vocab": len(tail_vocab),
        "filter_pairs": len(hr_to_tails),
        "tails_outside_vocab": appended_unknown,
        "protocol": "filtered",
    }


@torch.no_grad()
def evaluate(
    scorer: DistMultScorer,
    test_files: List[str],
    device: torch.device,
    *,
    protocol: str = "pooled",
    n_eval_triples: int = 2000,
    pool_size: int = 2000,
    filter_files: Optional[Sequence[str]] = None,
    max_filter_tails: Optional[int] = None,
    max_filter_rows: Optional[int] = None,
    batch_size: int = 64,
    k_hit: int = 10,
    max_text_len: int = 512,
    num_workers: int = 2,
    seed: int = 0,
) -> dict:
    """Compute MRR, Hit@k, nDCG over a streamed sample of test triples.

    ``protocol``:

    - ``"pooled"`` — rank against ``pool_size`` candidate tail strings
      sampled from ``test_files``. Fast and memory-bounded; useful during
      training and for quick sanity checks. (Default.)
    - ``"filtered"`` — rank against the *full* tail vocabulary built by
      streaming ``filter_files`` (defaults to ``test_files``; pass the
      union of train+valid+test for the standard KG-completion metric).
      Other known true tails for the same ``(h, r)`` are masked out.

    ``max_filter_tails`` caps the filtered-mode candidate vocabulary if
    you need to bound memory; it does not affect the filter-masking
    accuracy because the ``(h, r) -> {tails}`` index keeps growing.
    """
    scorer.eval()
    if not test_files:
        return {"MRR": 0.0, f"Hit@{k_hit}": 0.0, "nDCG": 0.0, "n": 0,
                "protocol": protocol}

    if protocol == "pooled":
        return _evaluate_pooled(
            scorer, test_files, device,
            n_eval_triples=n_eval_triples, pool_size=pool_size,
            batch_size=batch_size, k_hit=k_hit,
            max_text_len=max_text_len, num_workers=num_workers, seed=seed,
        )
    if protocol == "filtered":
        files = list(filter_files) if filter_files else list(test_files)
        return _evaluate_filtered(
            scorer, test_files, device,
            filter_files=files,
            n_eval_triples=n_eval_triples,
            batch_size=batch_size, k_hit=k_hit,
            max_text_len=max_text_len, num_workers=num_workers, seed=seed,
            max_tails=max_filter_tails,
            max_filter_rows=max_filter_rows,
        )
    raise ValueError(
        f"Unknown protocol {protocol!r} (expected one of {PROTOCOLS})"
    )


def _load_scorer_from_checkpoint(
    ckpt_path: str, device: torch.device
) -> DistMultScorer:
    """Reconstruct a DistMultScorer from a checkpoint saved by ``kgfm.train``."""
    from .encoders import make_encoder

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    encoder = make_encoder(
        cfg.get("encoder", "ngram"),
        vocab_size=cfg.get("vocab_size", 1 << 20),
        embedding_dim=cfg.get("embedding_dim", 256),
        n_min=cfg.get("n_min", 3),
        n_max=cfg.get("n_max", 5),
        max_ngrams=cfg.get("max_ngrams", 96),
        transformer_model=cfg.get("transformer_model", "bert-base-multilingual-cased"),
        transformer_max_length=cfg.get("transformer_max_length", 128),
        transformer_pooling=cfg.get("transformer_pooling", "mean"),
        freeze_encoder=cfg.get("freeze_encoder", False),
    )
    scorer = DistMultScorer(
        encoder, proj_dim=cfg.get("proj_dim"), normalize=True
    ).to(device)
    scorer.load_state_dict(ckpt["model_state"])
    scorer.eval()
    return scorer


def main() -> None:
    """CLI: ``kgfm-eval --ckpt PATH --test-list FILE [--protocol filtered ...]``."""
    import argparse

    from .data import discover_tsv_files, read_file_list, split_files_three_way
    from .utils import pick_free_gpu

    p = argparse.ArgumentParser(
        description="Evaluate a kgfm checkpoint with MRR / Hit@k / nDCG."
    )
    p.add_argument("--ckpt", required=True, help="Path to a checkpoint .pt file.")
    p.add_argument("--test-list", default=None,
                   help="Text file with one TSV path per line.")
    p.add_argument("--data-root", default="data",
                   help="Used when --test-list is omitted.")
    p.add_argument("--pattern", default="**/latest/*.tsv")
    p.add_argument("--test-buckets", type=int, default=1)
    p.add_argument("--n-buckets", type=int, default=10)
    p.add_argument("--protocol", default="pooled", choices=PROTOCOLS,
                   help="Ranking protocol. 'pooled' ranks against a small "
                        "candidate pool; 'filtered' ranks against the full "
                        "tail vocabulary with (h,r)->{tails} masking, "
                        "matching the standard KG-completion metric used "
                        "by ULTRA / MOTIF.")
    p.add_argument("--filter-list", action="append", default=None,
                   help="(filtered protocol only) Text file with TSV paths "
                        "used to build the candidate vocabulary and filter "
                        "index. Pass multiple times to combine train + "
                        "valid + test. Defaults to --test-list.")
    p.add_argument("--max-filter-tails", type=int, default=None,
                   help="(filtered protocol only) Cap the candidate "
                        "vocabulary size; the filter index keeps growing.")
    p.add_argument("--max-filter-rows", type=int, default=None,
                   help="(filtered protocol only) Cap total rows streamed "
                        "while building the filter index. Useful for "
                        "multi-TB corpora where exhausting the input is "
                        "impractical.")
    p.add_argument("--n-eval-triples", type=int, default=5000)
    p.add_argument("--pool-size", type=int, default=5000,
                   help="(pooled protocol only) Candidate pool size.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--k-hit", type=int, default=10)
    p.add_argument("--max-text-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.test_list:
        test_files = read_file_list(args.test_list)
    else:
        all_files = discover_tsv_files(args.data_root, args.pattern)
        _, _, test_files = split_files_three_way(
            all_files, valid_buckets=1,
            test_buckets=args.test_buckets, n_buckets=args.n_buckets,
        )

    if not test_files:
        raise SystemExit("No test files resolved.")

    filter_files = None
    if args.protocol == "filtered" and args.filter_list:
        filter_files = []
        for path in args.filter_list:
            filter_files.extend(read_file_list(path))

    device = pick_free_gpu()
    print(f"[eval] device={device}  protocol={args.protocol}  "
          f"test_files={len(test_files)}"
          + (f"  filter_files={len(filter_files)}" if filter_files else ""))

    scorer = _load_scorer_from_checkpoint(args.ckpt, device)
    metrics = evaluate(
        scorer, test_files, device,
        protocol=args.protocol,
        n_eval_triples=args.n_eval_triples,
        pool_size=args.pool_size,
        filter_files=filter_files,
        max_filter_tails=args.max_filter_tails,
        max_filter_rows=args.max_filter_rows,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        k_hit=args.k_hit,
        max_text_len=args.max_text_len,
        seed=args.seed,
    )
    print(f"[eval] {metrics}")


if __name__ == "__main__":
    main()
