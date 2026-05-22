"""Streaming training loop for relation prediction.

Train/valid/test splits are taken from text files (one TSV path per line) when
provided via --train-list / --valid-list / --test-list. If any of those is
omitted, files are auto-split by deterministic hash (default 80/10/10).

Encoders are pluggable: --encoder ngram (default) or --encoder transformer
(HuggingFace AutoModel; --transformer-model bert-base-multilingual-cased etc.).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    StreamingTripleDataset,
    collate_triples,
    discover_tsv_files,
    read_file_list,
    split_files_three_way,
)
from .encoders import make_encoder
from .eval import evaluate
from .model import DistMultScorer
from .utils import pick_free_gpu


@dataclass
class TrainConfig:
    # Data
    data_root: str = "data"
    pattern: str = "**/latest/*.tsv"
    train_list: Optional[str] = None
    valid_list: Optional[str] = None
    test_list: Optional[str] = None
    valid_buckets: int = 1
    test_buckets: int = 1
    n_buckets: int = 10
    # Encoder
    encoder: str = "ngram"
    vocab_size: int = 1 << 20
    embedding_dim: int = 256
    n_min: int = 3
    n_max: int = 5
    max_ngrams: int = 96
    transformer_model: str = "bert-base-multilingual-cased"
    transformer_max_length: int = 128
    transformer_pooling: str = "mean"
    freeze_encoder: bool = False
    proj_dim: Optional[int] = None
    # Loader
    max_text_len: int = 512
    batch_size: int = 256
    num_workers: int = 4
    shuffle_buffer: int = 16384
    row_keep_prob: float = 1.0
    max_rows_per_file: Optional[int] = None
    # Optim
    max_steps: int = 5000
    log_every: int = 50
    eval_every: int = 1000
    eval_pool_size: int = 2000
    eval_n_triples: int = 2000
    final_pool_size: int = 5000
    final_n_triples: int = 5000
    lr: float = 1e-3
    weight_decay: float = 0.0
    margin: float = 0.0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    # I/O
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 1000
    seed: int = 0
    use_bf16: bool = True
    max_files: Optional[int] = None
    cuda_visible: Optional[str] = None


def make_loader(files, cfg: TrainConfig, *, train: bool) -> DataLoader:
    ds = StreamingTripleDataset(
        files=files,
        shuffle_files=train,
        shuffle_buffer=cfg.shuffle_buffer if train else 1,
        row_keep_prob=cfg.row_keep_prob if train else 1.0,
        max_text_len=cfg.max_text_len,
        seed=cfg.seed,
        max_rows_per_file=cfg.max_rows_per_file,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_triples,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )


def in_batch_negative_loss(
    scorer: DistMultScorer,
    batch: dict,
    device: torch.device,
    *,
    margin: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """In-batch tail-corruption: score each (h_i, r_i) against all t_j."""
    h_text, r_text, t_text = batch["h_text"], batch["r_text"], batch["t_text"]
    h, r, t = scorer.encode_triple(h_text, r_text, t_text)
    h, r, t = h.to(device), r.to(device), t.to(device)

    hr = h * r  # [B, D]
    logits = hr @ t.t()  # [B, B]
    B = logits.size(0)
    target = torch.arange(B, device=device)

    if margin > 0.0:
        pos = logits.diag().unsqueeze(1)
        neg = logits.masked_fill(
            torch.eye(B, device=device, dtype=torch.bool), float("-inf")
        )
        return F.relu(margin - pos + neg).mean()
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)


def resolve_splits(cfg: TrainConfig) -> tuple[List[str], List[str], List[str]]:
    """Resolve the three file lists from CLI args, with auto-split fallback."""
    have_lists = any([cfg.train_list, cfg.valid_list, cfg.test_list])

    train: List[str] = []
    valid: List[str] = []
    test: List[str] = []

    if cfg.train_list:
        train = read_file_list(cfg.train_list)
    if cfg.valid_list:
        valid = read_file_list(cfg.valid_list)
    if cfg.test_list:
        test = read_file_list(cfg.test_list)

    if not have_lists:
        # No lists at all: auto-split everything under data_root.
        all_files = discover_tsv_files(cfg.data_root, cfg.pattern)
        if cfg.max_files is not None:
            all_files = all_files[: cfg.max_files]
        train, valid, test = split_files_three_way(
            all_files, cfg.valid_buckets, cfg.test_buckets, cfg.n_buckets
        )
        return train, valid, test

    # Some lists were provided. Auto-split for any missing one(s) using
    # only the files NOT named in the provided lists.
    if not (cfg.train_list and cfg.valid_list and cfg.test_list):
        named = set(train) | set(valid) | set(test)
        all_files = discover_tsv_files(cfg.data_root, cfg.pattern)
        remaining = [f for f in all_files if f not in named]
        auto_train, auto_valid, auto_test = split_files_three_way(
            remaining, cfg.valid_buckets, cfg.test_buckets, cfg.n_buckets
        )
        if not cfg.train_list:
            train = auto_train
        if not cfg.valid_list:
            valid = auto_valid
        if not cfg.test_list:
            test = auto_test

    if cfg.max_files is not None:
        train = train[: cfg.max_files]
    return train, valid, test


def train(cfg: TrainConfig) -> None:
    if cfg.cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible

    torch.manual_seed(cfg.seed)
    device = pick_free_gpu()
    print(f"[init] device={device}")

    train_files, valid_files, test_files = resolve_splits(cfg)
    print(
        f"[init] files: train={len(train_files)} "
        f"valid={len(valid_files)} test={len(test_files)}"
    )
    if not train_files:
        raise SystemExit("No training files found.")

    encoder = make_encoder(
        cfg.encoder,
        vocab_size=cfg.vocab_size,
        embedding_dim=cfg.embedding_dim,
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        max_ngrams=cfg.max_ngrams,
        transformer_model=cfg.transformer_model,
        transformer_max_length=cfg.transformer_max_length,
        transformer_pooling=cfg.transformer_pooling,
        freeze_encoder=cfg.freeze_encoder,
    )
    scorer = DistMultScorer(encoder, proj_dim=cfg.proj_dim, normalize=True).to(device)

    n_total = sum(p.numel() for p in scorer.parameters())
    n_trainable = sum(p.numel() for p in scorer.parameters() if p.requires_grad)
    print(
        f"[init] encoder={cfg.encoder} dim={scorer.dim} "
        f"params total={n_total:,} trainable={n_trainable:,}"
    )

    optim = torch.optim.AdamW(
        [p for p in scorer.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    autocast_dtype = torch.bfloat16 if (cfg.use_bf16 and device.type == "cuda") else None
    use_amp = autocast_dtype is not None

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    loader = make_loader(train_files, cfg, train=True)

    step = 0
    t0 = time.time()
    running = 0.0
    best_mrr = -1.0

    eval_target_files = valid_files if valid_files else test_files
    eval_label = "valid" if valid_files else "test"

    while step < cfg.max_steps:
        for batch in loader:
            step += 1
            scorer.train()
            optim.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    loss = in_batch_negative_loss(
                        scorer, batch, device,
                        margin=cfg.margin,
                        label_smoothing=cfg.label_smoothing,
                    )
            else:
                loss = in_batch_negative_loss(
                    scorer, batch, device,
                    margin=cfg.margin,
                    label_smoothing=cfg.label_smoothing,
                )
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg.grad_clip)
            optim.step()
            running += loss.item()

            if step % cfg.log_every == 0:
                dt = time.time() - t0
                avg = running / cfg.log_every
                running = 0.0
                print(
                    f"[step {step:>6d}] loss={avg:.4f}  "
                    f"rate={cfg.log_every*cfg.batch_size/dt:.0f} ex/s",
                    flush=True,
                )
                t0 = time.time()

            if step % cfg.eval_every == 0 and eval_target_files:
                metrics = evaluate(
                    scorer,
                    eval_target_files,
                    device,
                    n_eval_triples=cfg.eval_n_triples,
                    pool_size=cfg.eval_pool_size,
                    batch_size=max(64, cfg.batch_size // 2),
                    num_workers=max(1, cfg.num_workers // 2),
                    seed=cfg.seed,
                )
                print(f"[{eval_label} eval @ step {step}] {metrics}", flush=True)
                if metrics.get("MRR", 0.0) > best_mrr:
                    best_mrr = metrics["MRR"]
                    save_checkpoint(scorer, cfg, step, tag="best")
                t0 = time.time()

            if step % cfg.ckpt_every == 0:
                save_checkpoint(scorer, cfg, step, tag="last")

            if step >= cfg.max_steps:
                break

    save_checkpoint(scorer, cfg, step, tag="final")

    if test_files:
        # Reload best checkpoint for the final test report (if we had a valid set).
        if valid_files:
            best_path = os.path.join(cfg.ckpt_dir, "best.pt")
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device)
                scorer.load_state_dict(ckpt["model_state"])
                print(f"[test] loaded {best_path} (step={ckpt.get('step')})")
        final = evaluate(
            scorer,
            test_files,
            device,
            n_eval_triples=cfg.final_n_triples,
            pool_size=cfg.final_pool_size,
            batch_size=max(64, cfg.batch_size // 2),
            num_workers=max(1, cfg.num_workers // 2),
            seed=cfg.seed,
        )
        print(f"[final test] {final}", flush=True)


def save_checkpoint(scorer: nn.Module, cfg: TrainConfig, step: int, tag: str) -> None:
    path = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
    torch.save(
        {"step": step, "model_state": scorer.state_dict(), "config": cfg.__dict__},
        path,
    )
    print(f"[ckpt] saved {path} (step={step})")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data-root", default="data")
    p.add_argument("--pattern", default="**/latest/*.tsv")
    p.add_argument("--train-list", default=None,
                   help="Text file with one TSV path per line (training).")
    p.add_argument("--valid-list", default=None,
                   help="Text file with one TSV path per line (validation).")
    p.add_argument("--test-list", default=None,
                   help="Text file with one TSV path per line (test).")
    p.add_argument("--valid-buckets", type=int, default=1,
                   help="Hash buckets for auto-split valid (used if no --valid-list).")
    p.add_argument("--test-buckets", type=int, default=1,
                   help="Hash buckets for auto-split test (used if no --test-list).")
    p.add_argument("--n-buckets", type=int, default=10)
    # Encoder
    p.add_argument("--encoder", default="ngram",
                   choices=["ngram", "transformer", "bert", "hf"])
    p.add_argument("--vocab-size", type=int, default=1 << 20)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--n-min", type=int, default=3)
    p.add_argument("--n-max", type=int, default=5)
    p.add_argument("--max-ngrams", type=int, default=96)
    p.add_argument("--transformer-model", default="bert-base-multilingual-cased")
    p.add_argument("--transformer-max-length", type=int, default=128)
    p.add_argument("--transformer-pooling", default="mean", choices=["mean", "cls"])
    p.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze the LM and only train the projection head + r-bias.")
    p.add_argument("--proj-dim", type=int, default=None,
                   help="If set, add a Linear projection from encoder.embedding_dim "
                        "to proj_dim before scoring (recommended with frozen BERT).")
    # Loader
    p.add_argument("--max-text-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shuffle-buffer", type=int, default=16384)
    p.add_argument("--row-keep-prob", type=float, default=1.0)
    p.add_argument("--max-rows-per-file", type=int, default=None)
    # Optim
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-pool-size", type=int, default=2000)
    p.add_argument("--eval-n-triples", type=int, default=2000)
    p.add_argument("--final-pool-size", type=int, default=5000)
    p.add_argument("--final-n-triples", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # I/O
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--cuda-visible", default=None)

    a = p.parse_args()
    return TrainConfig(
        data_root=a.data_root,
        pattern=a.pattern,
        train_list=a.train_list,
        valid_list=a.valid_list,
        test_list=a.test_list,
        valid_buckets=a.valid_buckets,
        test_buckets=a.test_buckets,
        n_buckets=a.n_buckets,
        encoder=a.encoder,
        vocab_size=a.vocab_size,
        embedding_dim=a.embedding_dim,
        n_min=a.n_min,
        n_max=a.n_max,
        max_ngrams=a.max_ngrams,
        transformer_model=a.transformer_model,
        transformer_max_length=a.transformer_max_length,
        transformer_pooling=a.transformer_pooling,
        freeze_encoder=a.freeze_encoder,
        proj_dim=a.proj_dim,
        max_text_len=a.max_text_len,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        shuffle_buffer=a.shuffle_buffer,
        row_keep_prob=a.row_keep_prob,
        max_rows_per_file=a.max_rows_per_file,
        max_steps=a.max_steps,
        log_every=a.log_every,
        eval_every=a.eval_every,
        eval_pool_size=a.eval_pool_size,
        eval_n_triples=a.eval_n_triples,
        final_pool_size=a.final_pool_size,
        final_n_triples=a.final_n_triples,
        lr=a.lr,
        weight_decay=a.weight_decay,
        margin=a.margin,
        label_smoothing=a.label_smoothing,
        grad_clip=a.grad_clip,
        ckpt_dir=a.ckpt_dir,
        ckpt_every=a.ckpt_every,
        seed=a.seed,
        use_bf16=not a.no_bf16,
        max_files=a.max_files,
        cuda_visible=a.cuda_visible,
    )


def main() -> None:
    """CLI entrypoint: parse args and run training."""
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
