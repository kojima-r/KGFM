"""Streaming training loop for relation prediction.

Train/valid/test splits are taken from text files (one TSV path per line) when
provided via --train-list / --valid-list / --test-list. If any of those is
omitted, files are auto-split by deterministic hash (default 80/10/10).

Encoders are pluggable: --encoder ngram (default) or --encoder transformer
(HuggingFace AutoModel; --transformer-model bert-base-multilingual-cased etc.).
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
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
    # When set, overrides `batch_size` and is treated as the per-GPU
    # micro-batch size. Effective global batch size under DDP is
    # per_device_train_batch_size * world_size * gradient_accumulation_steps.
    per_device_train_batch_size: Optional[int] = None
    # When set, used as the per-GPU batch size for in-loop and final
    # evaluation. None falls back to max(64, per_device_train_batch_size // 2)
    # to preserve the existing single-GPU behaviour.
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    shuffle_buffer: int = 16384
    row_keep_prob: float = 1.0
    max_rows_per_file: Optional[int] = None
    # Distributed
    # Backend for torch.distributed when launched via torchrun. "nccl" is
    # used on CUDA, "gloo" otherwise. Only consulted when WORLD_SIZE>1.
    dist_backend: str = "nccl"
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
    # Resume
    # When True, train() looks in ckpt_dir for an existing checkpoint
    # (final.pt > last.pt > best.pt, in that order — "latest in time")
    # and continues training from its step counter up to max_steps.
    # Already-finished runs (step >= max_steps) skip the loop entirely
    # and fall through to the final-test eval pass.
    resume: bool = False
    # Explicit checkpoint path to resume from. Overrides the auto-detect
    # in ckpt_dir when set. Useful for branching off a specific ckpt.
    resume_from_ckpt: Optional[str] = None


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
    scorer: nn.Module,
    batch: dict,
    device: torch.device,
    *,
    margin: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """In-batch tail-corruption: score each (h_i, r_i) against all t_j.

    Accepts either a raw DistMultScorer or a DDP-wrapped one — we always
    encode through the underlying module. DDP's gradient all-reduce is
    registered on the parameters themselves, so backward still syncs
    correctly even though we bypass DDP.forward.
    """
    inner: DistMultScorer = scorer.module if hasattr(scorer, "module") else scorer
    h_text, r_text, t_text = batch["h_text"], batch["r_text"], batch["t_text"]
    h, r, t = inner.encode_triple(h_text, r_text, t_text)
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


@dataclass
class _DistState:
    """Holds the current torch.distributed state for the training process."""
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    enabled: bool = False  # True iff torch.distributed has been initialized
    device: torch.device = torch.device("cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _init_distributed(cfg: TrainConfig) -> _DistState:
    """Initialize torch.distributed if launched under torchrun, else no-op.

    Detection is purely env-var-based: WORLD_SIZE>1 means torchrun set us
    up. When that's true we init the process group, pin each rank to its
    LOCAL_RANK GPU, and return the full state. Otherwise we return a
    single-process state where the device is chosen by pick_free_gpu().
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return _DistState(device=pick_free_gpu())

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    backend = cfg.dist_backend
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return _DistState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        enabled=True,
        device=device,
    )


def _resolve_per_device_batch_size(cfg: TrainConfig) -> int:
    """Per-device micro-batch size. Honors per_device_train_batch_size when set."""
    if cfg.per_device_train_batch_size is not None:
        return int(cfg.per_device_train_batch_size)
    return int(cfg.batch_size)


def _resolve_eval_batch_size(cfg: TrainConfig, per_device_train_bs: int) -> int:
    """Per-device eval batch size. Honors per_device_eval_batch_size when set."""
    if cfg.per_device_eval_batch_size is not None:
        return int(cfg.per_device_eval_batch_size)
    # Original heuristic: half of the train batch, but at least 64.
    return max(64, per_device_train_bs // 2)


def train(cfg: TrainConfig) -> None:
    if cfg.cuda_visible is not None and "WORLD_SIZE" not in os.environ:
        # Under torchrun, CUDA_VISIBLE_DEVICES (if set) is honored by the
        # launcher; overriding it here would break the LOCAL_RANK->GPU
        # mapping. Only apply this knob in single-process mode.
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible

    torch.manual_seed(cfg.seed)
    ds = _init_distributed(cfg)
    device = ds.device

    def mprint(*args, **kwargs) -> None:
        """Print only from the main rank."""
        if ds.is_main:
            print(*args, **kwargs)

    per_device_bs = _resolve_per_device_batch_size(cfg)
    eval_bs = _resolve_eval_batch_size(cfg, per_device_bs)
    accum_steps = max(1, int(cfg.gradient_accumulation_steps))
    global_bs = per_device_bs * ds.world_size * accum_steps

    mprint(
        f"[init] device={device} world_size={ds.world_size} rank={ds.rank}"
        f" per_device_bs={per_device_bs} eval_bs={eval_bs} accum={accum_steps}"
        f" global_bs={global_bs}"
    )

    train_files, valid_files, test_files = resolve_splits(cfg)
    if ds.world_size > 1:
        # Shard train files across ranks. Slicing by rank::world_size keeps
        # the count balanced and the global ordering deterministic. Eval
        # files stay full because eval runs on rank 0 only.
        train_files = train_files[ds.rank :: ds.world_size]
    mprint(
        f"[init] files: train={len(train_files)} "
        f"valid={len(valid_files)} test={len(test_files)}"
    )
    if not train_files:
        raise SystemExit("No training files found for this rank.")

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
    scorer_raw = DistMultScorer(encoder, proj_dim=cfg.proj_dim, normalize=True).to(device)

    # ---- Resume: load model state BEFORE DDP wrap so DDP's initial
    # broadcast picks up the resumed weights as the rank-0 reference. ----
    resume_ckpt_path = _find_resume_ckpt(cfg)
    resumed_payload: Optional[dict] = None
    if resume_ckpt_path is not None:
        mprint(f"[resume] loading checkpoint: {resume_ckpt_path}")
        resumed_payload = torch.load(resume_ckpt_path, map_location=device)
        try:
            scorer_raw.load_state_dict(resumed_payload["model_state"])
        except RuntimeError as e:
            # Cross-config resumes (e.g. proj_dim changed) won't match
            # exactly; fall back to a non-strict load and warn.
            mprint(f"[resume] strict load failed ({e}); retrying non-strict")
            scorer_raw.load_state_dict(resumed_payload["model_state"], strict=False)

    n_total = sum(p.numel() for p in scorer_raw.parameters())
    n_trainable = sum(p.numel() for p in scorer_raw.parameters() if p.requires_grad)
    mprint(
        f"[init] encoder={cfg.encoder} dim={scorer_raw.dim} "
        f"params total={n_total:,} trainable={n_trainable:,}"
    )

    # Wrap with DDP only when there are trainable params to sync — DDP
    # complains if it has nothing to bucket. The frozen-encoder + no-proj
    # configuration produces a zero-trainable-param model.
    if ds.enabled and n_trainable > 0:
        device_ids = [ds.local_rank] if device.type == "cuda" else None
        scorer = DDP(scorer_raw, device_ids=device_ids)
    else:
        scorer = scorer_raw

    optim = torch.optim.AdamW(
        [p for p in scorer.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    autocast_dtype = torch.bfloat16 if (cfg.use_bf16 and device.type == "cuda") else None
    use_amp = autocast_dtype is not None

    if ds.is_main:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
    # Construct the DataLoader with the per-device micro-batch size.
    loader_cfg = TrainConfig(**{**cfg.__dict__, "batch_size": per_device_bs})
    loader = make_loader(train_files, loader_cfg, train=True)

    step = 0          # optimizer steps
    micro_done = 0    # micro-batches accumulated in the current optimizer step
    t0 = time.time()
    running = 0.0
    best_mrr = -1.0

    # ---- Resume: restore optimizer state + step counter + best_mrr. ----
    if resumed_payload is not None:
        if "optim_state" in resumed_payload:
            try:
                optim.load_state_dict(resumed_payload["optim_state"])
                mprint("[resume] loaded optimizer state")
            except (ValueError, RuntimeError) as e:
                mprint(
                    f"[resume] could not load optimizer state ({e}); "
                    "continuing with a fresh optimizer"
                )
        step = int(resumed_payload.get("step", 0))
        best_mrr = float(resumed_payload.get("best_mrr", -1.0))
        mprint(
            f"[resume] continuing from step={step}/{cfg.max_steps} "
            f"best_mrr={best_mrr:.4f}"
        )
        # Free the loaded payload — model + optimizer now hold the tensors.
        del resumed_payload
        if step >= cfg.max_steps:
            mprint(
                f"[resume] step {step} >= max_steps {cfg.max_steps}; "
                "skipping training loop and going straight to final eval"
            )

    eval_target_files = valid_files if valid_files else test_files
    eval_label = "valid" if valid_files else "test"

    def autocast_ctx():
        if use_amp:
            return torch.autocast(device_type="cuda", dtype=autocast_dtype)
        return contextlib.nullcontext()

    def sync_ctx(is_last_micro: bool):
        """DDP no_sync() while accumulating, then sync on the last micro."""
        if ds.enabled and not is_last_micro and hasattr(scorer, "no_sync"):
            return scorer.no_sync()
        return contextlib.nullcontext()

    def barrier() -> None:
        if ds.enabled:
            dist.barrier()

    optim.zero_grad(set_to_none=True)

    while step < cfg.max_steps:
        for batch in loader:
            scorer.train()
            is_last_micro = (micro_done + 1) == accum_steps
            with sync_ctx(is_last_micro):
                with autocast_ctx():
                    loss = in_batch_negative_loss(
                        scorer, batch, device,
                        margin=cfg.margin,
                        label_smoothing=cfg.label_smoothing,
                    )
                # Scale so the accumulated gradient corresponds to the mean
                # loss over the effective (global) batch.
                (loss / accum_steps).backward()
            running += loss.item()
            micro_done += 1

            if not is_last_micro:
                continue

            # End of an optimizer step.
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            micro_done = 0

            if step % cfg.log_every == 0:
                dt = time.time() - t0
                # `running` summed log_every * accum_steps micro-batch losses.
                avg = running / (cfg.log_every * accum_steps)
                running = 0.0
                # Throughput counts global examples seen since last log.
                ex = cfg.log_every * global_bs
                mprint(
                    f"[step {step:>6d}] loss={avg:.4f}  "
                    f"rate={ex/dt:.0f} ex/s",
                    flush=True,
                )
                t0 = time.time()

            if step % cfg.eval_every == 0 and eval_target_files:
                barrier()
                if ds.is_main:
                    metrics = evaluate(
                        scorer_raw,
                        eval_target_files,
                        device,
                        n_eval_triples=cfg.eval_n_triples,
                        pool_size=cfg.eval_pool_size,
                        batch_size=eval_bs,
                        num_workers=max(1, cfg.num_workers // 2),
                        seed=cfg.seed,
                    )
                    print(f"[{eval_label} eval @ step {step}] {metrics}", flush=True)
                    if metrics.get("MRR", 0.0) > best_mrr:
                        best_mrr = metrics["MRR"]
                        save_checkpoint(scorer_raw, cfg, step, tag="best",
                                        ds=ds, optim=optim, best_mrr=best_mrr)
                barrier()
                t0 = time.time()

            if step % cfg.ckpt_every == 0:
                save_checkpoint(scorer_raw, cfg, step, tag="last",
                                ds=ds, optim=optim, best_mrr=best_mrr)

            if step >= cfg.max_steps:
                break

    save_checkpoint(scorer_raw, cfg, step, tag="final",
                    ds=ds, optim=optim, best_mrr=best_mrr)

    if test_files:
        # Reload best checkpoint for the final test report (if we had a valid set).
        barrier()
        if ds.is_main:
            if valid_files:
                best_path = os.path.join(cfg.ckpt_dir, "best.pt")
                if os.path.exists(best_path):
                    ckpt = torch.load(best_path, map_location=device)
                    scorer_raw.load_state_dict(ckpt["model_state"])
                    print(f"[test] loaded {best_path} (step={ckpt.get('step')})")
            final = evaluate(
                scorer_raw,
                test_files,
                device,
                n_eval_triples=cfg.final_n_triples,
                pool_size=cfg.final_pool_size,
                batch_size=eval_bs,
                num_workers=max(1, cfg.num_workers // 2),
                seed=cfg.seed,
            )
            print(f"[final test] {final}", flush=True)
        barrier()

    if ds.enabled:
        dist.destroy_process_group()


def save_checkpoint(
    scorer: nn.Module,
    cfg: TrainConfig,
    step: int,
    tag: str,
    ds: Optional["_DistState"] = None,
    optim: Optional[torch.optim.Optimizer] = None,
    best_mrr: Optional[float] = None,
) -> None:
    """Save a checkpoint. In distributed mode, only rank 0 writes.

    Saves the optimizer state and best validation MRR so a later run with
    ``resume=True`` can pick up exactly where this one left off (same
    optimizer momentum, same "best" target). Both are optional so the
    save still works during inference-only paths.
    """
    if ds is not None and not ds.is_main:
        return
    # If a DDP-wrapped module slipped through, unwrap so the state-dict
    # keys don't carry the "module." prefix.
    state = scorer.module.state_dict() if hasattr(scorer, "module") else scorer.state_dict()
    path = os.path.join(cfg.ckpt_dir, f"{tag}.pt")
    payload: dict = {"step": step, "model_state": state, "config": cfg.__dict__}
    if optim is not None:
        payload["optim_state"] = optim.state_dict()
    if best_mrr is not None:
        payload["best_mrr"] = float(best_mrr)
    torch.save(payload, path)
    print(f"[ckpt] saved {path} (step={step})")


def _find_resume_ckpt(cfg: TrainConfig) -> Optional[str]:
    """Locate the checkpoint to resume from, if any.

    Explicit ``resume_from_ckpt`` wins; otherwise we pick the most
    recent-in-time snapshot from ``ckpt_dir``: ``final.pt`` (training
    completed) > ``last.pt`` (most recent --ckpt-every snapshot) >
    ``best.pt`` (highest validation MRR, possibly older).
    """
    if cfg.resume_from_ckpt:
        return cfg.resume_from_ckpt if os.path.exists(cfg.resume_from_ckpt) else None
    if not cfg.resume:
        return None
    for tag in ("final.pt", "last.pt", "best.pt"):
        candidate = os.path.join(cfg.ckpt_dir, tag)
        if os.path.exists(candidate):
            return candidate
    return None


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
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-device micro-batch size. Kept for backwards "
                        "compatibility; --per-device-train-batch-size "
                        "overrides this when set.")
    p.add_argument("--per-device-train-batch-size", type=int, default=None,
                   help="HF-style per-GPU micro-batch size. Effective "
                        "global batch size = "
                        "per_device_train_batch_size * world_size * "
                        "gradient_accumulation_steps.")
    p.add_argument("--per-device-eval-batch-size", type=int, default=None,
                   help="HF-style per-GPU eval batch size, used for in-loop "
                        "validation and the final test pass. Defaults to "
                        "max(64, per_device_train_batch_size // 2).")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Number of forward/backward passes per optimizer "
                        "step. Combined with DDP and "
                        "--per-device-train-batch-size to control the "
                        "effective global batch size.")
    p.add_argument("--dist-backend", default="nccl",
                   help="torch.distributed backend used under torchrun "
                        "(falls back to gloo on CPU).")
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
    p.add_argument("--resume", action="store_true",
                   help="Auto-detect a checkpoint in --ckpt-dir "
                        "(final.pt > last.pt > best.pt) and continue "
                        "training from its saved step up to --max-steps. "
                        "If the saved step already reached --max-steps, "
                        "the training loop is skipped entirely.")
    p.add_argument("--resume-from-ckpt", default=None,
                   help="Explicit checkpoint path to resume from. "
                        "Overrides the auto-detect inside --ckpt-dir.")

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
        per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_eval_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        dist_backend=a.dist_backend,
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
        resume=a.resume,
        resume_from_ckpt=a.resume_from_ckpt,
    )


def main() -> None:
    """CLI entrypoint: parse args and run training."""
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
