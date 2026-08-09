"""One kgfm benchmark cell: train on ChEMBL, evaluate, write a JSON record.

kgfm is not a foundation model — there is no pretrained ChEMBL checkpoint to
do zero-shot inference with — so a cell *trains* on ``list_chembl/train.txt``
and *evaluates* on ``list_chembl/test.txt``. ULTRA / MOTIF are the zero-shot
baselines it is compared against; see benchmarks/README.md for the resulting
protocol caveats.

This runs as its own process (see `sweep.py`): training touches global CUDA
and torch.distributed state, and multi-GPU cells are launched under torchrun,
which needs a process to launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch

from ..data import read_file_list
from ..encoders import make_encoder
from ..eval import evaluate
from ..model import DistMultScorer
from ..losses import DEFAULT_LOSS, DEFAULT_TEMPERATURE, LOSSES
from ..train import TrainConfig, train as kgfm_train
from ..utils import pick_free_gpu


def is_main_rank() -> bool:
    """True on rank 0 under torchrun, and in single-process runs."""
    return int(os.environ.get("RANK", "0")) == 0


def _eval_batch_size(args: argparse.Namespace) -> int:
    if args.per_device_eval_batch_size:
        return int(args.per_device_eval_batch_size)
    train_bs = args.per_device_train_batch_size or args.batch_size
    return max(64, train_bs // 2)


def _build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        train_list=args.train_list,
        valid_list=args.valid_list,
        test_list=args.test_list,
        encoder=args.encoder,
        embedding_dim=args.embedding_dim,
        proj_dim=args.proj_dim,
        transformer_model=args.transformer_model,
        freeze_encoder=args.freeze_encoder,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_pool_size=args.pool_size,
        valid_loss_batches=args.valid_loss_batches,
        eval_n_triples=args.n_eval_triples,
        final_pool_size=args.pool_size,
        final_n_triples=args.n_eval_triples,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        lr=args.lr,
        loss=args.loss,
        loss_temperature=args.loss_temperature,
        weight_decay=args.weight_decay,
        encoder_weight_decay=args.encoder_weight_decay,
        head_weight_decay=args.head_weight_decay,
        encoder_dropout=args.encoder_dropout,
        head_dropout=args.head_dropout,
        mask_duplicate_tails=args.mask_duplicate_tails,
        seed=args.seed,
        resume=args.resume,
        resume_from_ckpt=args.resume_from_ckpt,
    )


def _final_eval(ckpt_path: str, args: argparse.Namespace) -> dict:
    """Reload the selected checkpoint and run a fresh evaluation pass.

    The encoder is rebuilt from the config stored *in the checkpoint*, not
    from the CLI, so a re-scoring pass can never silently disagree with how
    the weights were trained. (kgfm/eval.py does the same for `kgfm eval`;
    a new encoder argument has to be threaded through both.)
    """
    # Under torchrun, pin to LOCAL_RANK's GPU instead of racing pick_free_gpu().
    if "LOCAL_RANK" in os.environ and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    else:
        device = pick_free_gpu()

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {}) or {}
    encoder = make_encoder(
        cfg.get("encoder", args.encoder),
        vocab_size=cfg.get("vocab_size", 1 << 20),
        embedding_dim=cfg.get("embedding_dim", args.embedding_dim),
        n_min=cfg.get("n_min", 3),
        n_max=cfg.get("n_max", 5),
        max_ngrams=cfg.get("max_ngrams", 96),
        transformer_model=cfg.get("transformer_model", args.transformer_model),
        transformer_max_length=cfg.get("transformer_max_length", 128),
        transformer_pooling=cfg.get("transformer_pooling", "mean"),
        freeze_encoder=cfg.get("freeze_encoder", args.freeze_encoder),
        encoder_dropout=cfg.get("encoder_dropout"),
    )
    scorer = DistMultScorer(
        encoder, proj_dim=cfg.get("proj_dim", args.proj_dim), normalize=True,
        head_dropout=cfg.get("head_dropout", 0.0),
    ).to(device)
    scorer.load_state_dict(ckpt["model_state"])

    test_files = read_file_list(args.test_list)

    filter_files = None
    if args.protocol == "filtered":
        # Build the filter index from train+valid+test by default so the
        # filtered metric matches the standard KG-completion definition.
        filter_lists = args.filter_list or [
            args.train_list, args.valid_list, args.test_list,
        ]
        filter_files = []
        for path in filter_lists:
            if path:
                filter_files.extend(read_file_list(path))

    return evaluate(
        scorer, test_files, device,
        protocol=args.protocol,
        n_eval_triples=args.n_eval_triples,
        pool_size=args.pool_size,
        filter_files=filter_files,
        max_filter_tails=args.max_filter_tails,
        max_filter_rows=args.max_filter_rows,
        batch_size=_eval_batch_size(args),
        num_workers=max(1, args.num_workers // 2),
        seed=args.seed,
    )


def _select_checkpoint(ckpt_dir: str) -> Optional[str]:
    """best (highest valid MRR) > final (end of run) > last (periodic snapshot).

    The fall-through to last.pt matters when resuming an interrupted run that
    wrote checkpoints but never reached validation or completion.
    """
    for tag in ("best.pt", "final.pt", "last.pt"):
        candidate = os.path.join(ckpt_dir, tag)
        if os.path.exists(candidate):
            return candidate
    return None


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-list", default="list_chembl/train.txt")
    p.add_argument("--valid-list", default="list_chembl/valid.txt")
    p.add_argument("--test-list", default="list_chembl/test.txt")
    p.add_argument("--encoder", default="ngram",
                   choices=["ngram", "transformer", "bert", "hf"])
    p.add_argument("--transformer-model", default="bert-base-multilingual-cased")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-device micro-batch size.")
    p.add_argument("--per-device-train-batch-size", type=int, default=None,
                   help="HF-style per-GPU micro-batch size. Effective global "
                        "batch = this * world_size * grad-accum.")
    p.add_argument("--per-device-eval-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--valid-loss-batches", type=int, default=10)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate; default is chosen per encoder.")
    p.add_argument("--loss", default=DEFAULT_LOSS, choices=list(LOSSES))
    p.add_argument("--loss-temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--encoder-weight-decay", type=float, default=None)
    p.add_argument("--head-weight-decay", type=float, default=None)
    p.add_argument("--encoder-dropout", type=float, default=None)
    p.add_argument("--head-dropout", type=float, default=0.0)
    p.add_argument("--no-mask-duplicate-tails", dest="mask_duplicate_tails",
                   action="store_false", default=True)
    p.add_argument("--n-eval-triples", type=int, default=5000)
    p.add_argument("--pool-size", type=int, default=5000)
    p.add_argument("--protocol", default="pooled", choices=["pooled", "filtered"],
                   help="Final-eval protocol. In-loop validation during "
                        "training is always pooled regardless.")
    p.add_argument("--filter-list", action="append", default=None,
                   help="(filtered only) TSV path lists for the candidate "
                        "vocabulary and (h,r)->{tails} index. Defaults to "
                        "train+valid+test.")
    p.add_argument("--max-filter-tails", type=int, default=None)
    p.add_argument("--max-filter-rows", type=int, default=None)
    p.add_argument("--ckpt-dir", default="checkpoints/bench_chembl_kgfm")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmarks/results/kgfm.json")
    p.add_argument("--skip-train", action="store_true",
                   help="Only run the final eval on the existing checkpoint.")
    p.add_argument("--resume", action="store_true",
                   help="Continue from a checkpoint in --ckpt-dir "
                        "(final.pt > last.pt > best.pt) up to --max-steps.")
    p.add_argument("--resume-from-ckpt", default=None)


def run_from_args(args: argparse.Namespace) -> Optional[dict]:
    if is_main_rank():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    if not args.skip_train:
        kgfm_train(_build_config(args))
    train_seconds = time.time() - t0

    # Final eval + JSON write are inherently single-process: the checkpoint is
    # identical on every rank, so other ranks would just race on the output.
    if not is_main_rank():
        return None

    ckpt = _select_checkpoint(args.ckpt_dir)
    if ckpt is None:
        sys.exit(f"No checkpoint found under {args.ckpt_dir}.")

    metrics = _final_eval(ckpt, args)

    protocol_meta = {
        "type": args.protocol,
        "n_eval_triples": args.n_eval_triples,
        "k_hit": 10,
    }
    if args.protocol == "pooled":
        protocol_meta["pool_size"] = args.pool_size
    else:
        protocol_meta["max_filter_tails"] = args.max_filter_tails
        protocol_meta["filter_list"] = args.filter_list

    record = {
        "method": "kgfm",
        "encoder": args.encoder,
        "freeze_encoder": bool(args.freeze_encoder),
        "proj_dim": args.proj_dim,
        "batch_size": args.batch_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "ckpt": ckpt,
        "train_seconds": None if args.skip_train else round(train_seconds, 1),
        "metrics": metrics,
        "protocol": protocol_meta,
        "data": {
            "train_list": args.train_list,
            "valid_list": args.valid_list,
            "test_list": args.test_list,
        },
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[cell] wrote {args.out}")
    print(json.dumps(record, indent=2))
    return record
