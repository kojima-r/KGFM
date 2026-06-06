"""Train kgfm on list_chembl and dump test metrics as JSON.

Note that kgfm is not a foundation model — there is no pretrained checkpoint
to do "zero-shot" inference with on chembl. We therefore *train* on
list_chembl/train.txt and *evaluate* on list_chembl/test.txt. ULTRA / MOTIF
are the zero-shot baselines they are compared against in this benchmark
suite; see ``benchmarks/README.md`` for the protocol caveats.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow running this script directly without `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from kgfm.data import read_file_list  # noqa: E402
from kgfm.encoders import make_encoder  # noqa: E402
from kgfm.eval import evaluate  # noqa: E402
from kgfm.model import DistMultScorer  # noqa: E402
from kgfm.train import TrainConfig, train as kgfm_train  # noqa: E402
from kgfm.utils import pick_free_gpu  # noqa: E402


def _is_main_rank() -> bool:
    """True for the rank-0 process under torchrun, else True (single-process)."""
    return int(os.environ.get("RANK", "0")) == 0


def _eval_batch_size(args) -> int:
    """Per-device eval batch size, defaulting to max(64, train_bs // 2)."""
    if getattr(args, "per_device_eval_batch_size", None):
        return int(args.per_device_eval_batch_size)
    train_bs = getattr(args, "per_device_train_batch_size", None) or args.batch_size
    return max(64, train_bs // 2)


def _build_config(args) -> TrainConfig:
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
        eval_n_triples=args.n_eval_triples,
        final_pool_size=args.pool_size,
        final_n_triples=args.n_eval_triples,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        seed=args.seed,
        resume=args.resume,
        resume_from_ckpt=args.resume_from_ckpt,
    )


def _final_eval(ckpt_path: str, test_list: str, args) -> dict:
    """Reload the best checkpoint and run a fresh evaluation pass."""
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
    )
    scorer = DistMultScorer(
        encoder, proj_dim=cfg.get("proj_dim", args.proj_dim), normalize=True,
    ).to(device)
    scorer.load_state_dict(ckpt["model_state"])

    test_files = read_file_list(test_list)

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-list", default="list_chembl/train.txt")
    p.add_argument("--valid-list", default="list_chembl/valid.txt")
    p.add_argument("--test-list",  default="list_chembl/test.txt")
    p.add_argument("--encoder", default="ngram",
                   choices=["ngram", "transformer", "bert", "hf"])
    p.add_argument("--transformer-model", default="bert-base-multilingual-cased")
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256,
                   help="Per-device micro-batch size. Backwards-compatible "
                        "alias for --per-device-train-batch-size.")
    p.add_argument("--per-device-train-batch-size", type=int, default=None,
                   help="HF-style per-GPU micro-batch size. Effective global "
                        "batch size = per_device_train_batch_size * "
                        "world_size * gradient_accumulation_steps.")
    p.add_argument("--per-device-eval-batch-size", type=int, default=None,
                   help="Per-GPU eval batch size. Used by both in-loop "
                        "validation and the final eval. Defaults to "
                        "max(64, per_device_train_batch_size // 2).")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="Number of forward/backward passes per optimizer "
                        "step under DDP. Multiplied with world_size and "
                        "per_device_train_batch_size to get the effective "
                        "global batch size.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--n-eval-triples", type=int, default=5000)
    p.add_argument("--pool-size", type=int, default=5000)
    p.add_argument("--protocol", default="pooled",
                   choices=["pooled", "filtered"],
                   help="Final-eval protocol. 'filtered' matches the "
                        "standard KG-completion metric used by ULTRA / "
                        "MOTIF. In-loop validation during training is "
                        "always pooled regardless.")
    p.add_argument("--filter-list", action="append", default=None,
                   help="(filtered protocol only) TSV path lists used to "
                        "build the candidate vocabulary and (h,r)->{tails} "
                        "filter index. Pass multiple times. Defaults to "
                        "the union of --train-list / --valid-list / "
                        "--test-list.")
    p.add_argument("--max-filter-tails", type=int, default=None,
                   help="(filtered protocol only) Cap the candidate "
                        "vocabulary size; the filter index keeps growing.")
    p.add_argument("--max-filter-rows", type=int, default=None,
                   help="(filtered protocol only) Cap total rows streamed "
                        "while building the filter index. Strongly "
                        "recommended on chembl-scale corpora.")
    p.add_argument("--ckpt-dir", default="checkpoints/bench_chembl_kgfm")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="benchmarks/results/kgfm.json")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; only run final eval on the existing best ckpt.")
    p.add_argument("--resume", action="store_true",
                   help="If --ckpt-dir already contains a checkpoint "
                        "(final.pt > last.pt > best.pt), load it and "
                        "continue training from its saved step up to "
                        "--max-steps. A no-op when no checkpoint is "
                        "present, and a training-skip when the saved "
                        "step has already reached --max-steps.")
    p.add_argument("--resume-from-ckpt", default=None,
                   help="Explicit checkpoint path to resume from. "
                        "Overrides the --resume auto-detect.")
    args = p.parse_args()

    if _is_main_rank():
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    if not args.skip_train:
        cfg = _build_config(args)
        kgfm_train(cfg)
    train_seconds = time.time() - t0

    # Final eval + JSON write are inherently single-process: the checkpoint
    # produced by training is identical on every rank, so re-running eval on
    # all ranks would just race on the output file.
    if not _is_main_rank():
        return

    # Preference order: best (highest valid MRR) > final (end of run) >
    # last (most recent ckpt_every snapshot). The fall-through to last.pt
    # matters for resuming an interrupted run that wrote ckpts but never
    # got to validation / training completion.
    best_ckpt = None
    for tag in ("best.pt", "final.pt", "last.pt"):
        candidate = os.path.join(args.ckpt_dir, tag)
        if os.path.exists(candidate):
            best_ckpt = candidate
            break
    if best_ckpt is None:
        sys.exit(f"No checkpoint found under {args.ckpt_dir}.")

    metrics = _final_eval(best_ckpt, args.test_list, args)

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

    out = {
        "method": "kgfm",
        "encoder": args.encoder,
        "freeze_encoder": bool(args.freeze_encoder),
        "proj_dim": args.proj_dim,
        "batch_size": args.batch_size,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "ckpt": best_ckpt,
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
        json.dump(out, f, indent=2)
    print(f"[run_kgfm] wrote {args.out}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
