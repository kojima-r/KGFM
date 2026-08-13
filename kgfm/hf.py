"""Publish a trained scorer to the HuggingFace Hub (`kgfm hf`).

A kgfm checkpoint is not a HuggingFace model: it is a `torch.save` of
``{step, model_state, config, optim_state, best_mrr}`` where ``config`` is a
``TrainConfig``. Nothing on the Hub knows how to rebuild that, so what gets
published is the payload plus enough metadata for ``kgfm.hf.load`` (and a
human reading the model card) to reconstruct it exactly.

Two things are stripped or split out, both of which matter at this scale:

**The optimizer state is dropped by default.** It is two moments per
parameter, so it is roughly twice the size of the model and is useless for
inference — an ngram checkpoint goes from 3.1 GB to 1.1 GB. ``--with-optimizer``
keeps it for anyone who wants to resume training from the Hub.

**A frozen encoder is not uploaded at all** (``--mode head-only``, the default
for those checkpoints). ``freeze_encoder=True`` means the LM's parameters had
``requires_grad=False`` for the whole run, so the weights in the checkpoint are
bit-identical to the public model they came from; re-uploading them is a
gigabyte-scale copy of someone else's release. Measured on this repo's own
runs: a frozen ``bge-large`` cell is 1.05 MB of head against 1.34 GB of
encoder, and a frozen ``e5-mistral-7b`` checkpoint directory is 40 GB. The head
alone is the entire trained artifact. ``load`` rebuilds the encoder from its
preset and loads the head over it.

Head-only is refused when it would be lossy — a fine-tuned encoder, or the
ngram encoder, which is trained from scratch and has no public copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .encoders import ENCODER_PRESETS, is_transformer, preset_info

MODEL_FILE = "kgfm_model.pt"
CONFIG_FILE = "kgfm_config.json"
CARD_FILE = "README.md"
UPLOAD_MODES = ("auto", "full", "head-only")
CKPT_CHOICES = ("best", "final", "last")


# ---------------------------------------------------------------------------
# Locating the checkpoint and its metrics
# ---------------------------------------------------------------------------


def resolve_checkpoint(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    """(checkpoint path, run directory or None).

    Either ``--ckpt`` directly, or ``--out-dir`` + ``--tag``, which mirrors how
    the rest of the CLI addresses a benchmark run.
    """
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        # `<run>/kgfm_ckpts_<tag>/best.pt` -> `<run>`, so the metrics JSON and
        # the run metadata can be picked up without being asked for.
        run_dir = path.parent.parent if path.parent.name.startswith("kgfm_ckpts_") else None
        return path, run_dir

    if not args.tag:
        raise SystemExit(
            "Pass --ckpt PATH, or --out-dir RUN --tag CELL to take a "
            "checkpoint from a benchmark run."
        )
    from .runs import resolve_run_dir

    if not args.out_dir:
        raise SystemExit("--tag needs --out-dir (e.g. --out-dir latest).")
    # (results_root, target) — the same order report.py uses.
    run_dir = Path(resolve_run_dir(args.results_root, args.out_dir))
    ckpt_dir = run_dir / f"kgfm_ckpts_{args.tag}"
    if not ckpt_dir.is_dir():
        available = sorted(p.name.replace("kgfm_ckpts_", "")
                           for p in run_dir.glob("kgfm_ckpts_*"))
        raise SystemExit(
            f"No checkpoints for tag {args.tag!r} in {run_dir}\n"
            f"Available tags: {', '.join(available) or '(none)'}"
        )
    path = ckpt_dir / f"{args.which}.pt"
    if not path.is_file():
        present = sorted(p.name for p in ckpt_dir.glob("*.pt"))
        raise SystemExit(
            f"{path} does not exist (present: {', '.join(present) or 'none'})"
        )
    return path, run_dir


def find_metrics(run_dir: Optional[Path], tag: Optional[str]) -> Dict[str, Any]:
    """Final metrics for this cell, keyed by protocol, if the run has them."""
    if run_dir is None or not tag:
        return {}
    out: Dict[str, Any] = {}
    for path in sorted(run_dir.glob(f"kgfm_*_{tag}.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Guard against `kgfm_pooled_ngram.json` matching tag "ngram" when the
        # cell is really "ngram_linear": compare the recorded identity instead.
        proto = (rec.get("protocol") or {}).get("type")
        if proto and rec.get("metrics"):
            out[proto] = {"metrics": rec["metrics"], "protocol": rec["protocol"]}
    return out


# ---------------------------------------------------------------------------
# Building what gets uploaded
# ---------------------------------------------------------------------------


def _split_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(encoder tensors, head tensors) from a scorer state dict."""
    enc = {k: v for k, v in state.items() if k.startswith("encoder.")}
    head = {k: v for k, v in state.items() if not k.startswith("encoder.")}
    return enc, head


def resolve_mode(mode: str, cfg: Dict[str, Any]) -> str:
    """Turn ``auto`` into a concrete mode, and reject impossible ones.

    head-only is only lossless when the encoder is a pretrained LM that was
    frozen for the whole run: then its weights are still the public ones and
    `load` can fetch them again.
    """
    encoder = str(cfg.get("encoder", ""))
    frozen = bool(cfg.get("freeze_encoder"))
    reusable = frozen and is_transformer(encoder)
    if mode == "auto":
        return "head-only" if reusable else "full"
    if mode == "head-only" and not reusable:
        why = ("the encoder was fine-tuned, so its weights are no longer the "
               "public ones" if is_transformer(encoder) else
               f"the {encoder!r} encoder is trained from scratch and has no "
               "public copy to rebuild from")
        raise SystemExit(f"--mode head-only would lose the model: {why}.")
    return mode


def build_payload(ckpt: Dict[str, Any], mode: str, with_optimizer: bool) -> Dict[str, Any]:
    """The dict that gets torch.save'd into the repo."""
    state = ckpt["model_state"]
    if mode == "head-only":
        _, state = _split_state(state)
    payload: Dict[str, Any] = {
        "model_state": state,
        "config": ckpt.get("config", {}),
        "step": ckpt.get("step"),
        "kgfm_upload_mode": mode,
    }
    if ckpt.get("best_mrr") is not None:
        payload["best_mrr"] = ckpt["best_mrr"]
    if with_optimizer and "optim_state" in ckpt:
        payload["optim_state"] = ckpt["optim_state"]
    return payload


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def _state_bytes(state: Dict[str, Any]) -> int:
    return sum(v.numel() * v.element_size() for v in state.values()
               if hasattr(v, "numel"))


def model_card(
    cfg: Dict[str, Any],
    mode: str,
    step: Optional[int],
    metrics: Dict[str, Any],
    repo_id: str,
    base_model: Optional[str],
) -> str:
    """The repo's README.md, including the YAML frontmatter the Hub reads."""
    encoder = cfg.get("encoder", "?")
    head = cfg.get("head", "auto")
    tags = ["knowledge-graph", "link-prediction", "distmult", "kgfm"]
    front = [
        "---",
        "library_name: kgfm",
        "pipeline_tag: feature-extraction",
        "tags:",
        *[f"  - {t}" for t in tags],
    ]
    if base_model:
        front += ["base_model:", f"  - {base_model}"]
    front += ["---", ""]

    rows = [
        ("encoder", encoder),
        ("head", head),
        ("proj_dim (scoring width)", cfg.get("proj_dim")),
        ("freeze_encoder", cfg.get("freeze_encoder")),
        ("loss", cfg.get("loss")),
        ("loss_temperature", cfg.get("loss_temperature")),
        ("batch size (= negatives + 1)", cfg.get("batch_size")),
        ("learning rate", cfg.get("lr") or "per-encoder default"),
        ("steps trained", step),
    ]
    body = [
        f"# {repo_id}",
        "",
        "A `kgfm` DistMult scorer over a **text encoder**:",
        "`score(h, r, t) = Σ_d h_d · r_d · t_d`, where `h`, `r` and `t` are",
        "embeddings of raw strings rather than rows of an entity table. There is",
        "no entity vocabulary, so unseen entities work by construction.",
        "",
        "## Architecture",
        "",
        "| setting | value |",
        "|---|---|",
        *[f"| {k} | `{v}` |" for k, v in rows],
        "",
    ]

    if mode == "head-only":
        body += [
            "## Contents",
            "",
            f"**This repo contains the projection head only.** The encoder",
            f"(`{base_model}`) was frozen for the whole run, so its weights are",
            "still bit-identical to the public release and are not duplicated",
            "here — `kgfm.hf.load` downloads them from that repo and loads this",
            "head over them. The head is the entire trained artifact.",
            "",
        ]
    else:
        body += [
            "## Contents",
            "",
            "The full scorer: encoder weights and projection head.",
            "",
        ]

    if metrics:
        body += ["## Evaluation", "", "| protocol | MRR | Hit@10 | nDCG | n |",
                 "|---|---|---|---|---|"]
        for proto, blob in sorted(metrics.items()):
            m = blob["metrics"]
            def g(key):
                v = m.get(key)
                return f"{v:.4f}" if isinstance(v, (int, float)) else "—"
            body.append(f"| {proto} | {g('MRR')} | {g('Hit@10')} | {g('nDCG')} "
                        f"| {m.get('n', '—')} |")
        body += [
            "",
            "`filtered` ranks against the full tail vocabulary with other known",
            "true tails masked, and is the protocol to use for comparison with",
            "other KG models. `pooled` ranks against a sampled candidate pool and",
            "is cheaper but easier, so its numbers are not comparable to",
            "published filtered results.",
            "",
        ]

    body += [
        "## Usage",
        "",
        "```python",
        "from kgfm.hf import load",
        "",
        f'scorer = load("{repo_id}")          # downloads and rebuilds',
        'h, r, t = scorer.encode_triple(["aspirin"], ["treats"], ["headache"])',
        "score = scorer.score(h, r, t)",
        "```",
        "",
    ]
    if mode == "head-only":
        # `kgfm eval --ckpt` loads a state dict strictly, so it cannot read a
        # head-only payload — pointing people at it would just hand them a
        # missing-keys error.
        body += [
            "`kgfm eval --ckpt` will **not** read this file directly: it loads a",
            "full state dict, and the encoder tensors are deliberately absent.",
            "Use `load` above, or publish with `--mode full` if you need a",
            "self-contained checkpoint.",
            "",
        ]
    else:
        body += [
            "The payload is a self-contained kgfm checkpoint, so it also works",
            "with the CLI directly:",
            "",
            "```bash",
            "kgfm eval --ckpt kgfm_model.pt --test-list list_chembl/test.txt",
            "```",
            "",
        ]
    return "\n".join(front + body)


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def _stage(
    staging: Path,
    ckpt: Dict[str, Any],
    cfg: Dict[str, Any],
    mode: str,
    with_optimizer: bool,
    metrics: Dict[str, Any],
    repo_id: str,
) -> Dict[str, int]:
    """Write the files to upload into ``staging``; return their sizes."""
    payload = build_payload(ckpt, mode, with_optimizer)
    base_model = preset_info(str(cfg.get("encoder", ""))).get("model")
    if mode == "head-only" and not base_model:
        # `--encoder transformer --transformer-model X` has no preset entry.
        base_model = cfg.get("transformer_model")

    torch.save(payload, staging / MODEL_FILE)
    (staging / CONFIG_FILE).write_text(json.dumps({
        "kgfm_upload_mode": mode,
        "base_model": base_model,
        "step": ckpt.get("step"),
        "best_mrr": ckpt.get("best_mrr"),
        "metrics": {k: v["metrics"] for k, v in metrics.items()},
        "train_config": cfg,
    }, indent=2, default=str) + "\n")
    (staging / CARD_FILE).write_text(
        model_card(cfg, mode, ckpt.get("step"), metrics, repo_id, base_model)
    )
    return {p.name: p.stat().st_size for p in sorted(staging.iterdir())}


def run_from_args(args: argparse.Namespace) -> None:
    ckpt_path, run_dir = resolve_checkpoint(args)
    tag = args.tag or (ckpt_path.parent.name.replace("kgfm_ckpts_", "")
                       if ckpt_path.parent.name.startswith("kgfm_ckpts_") else None)

    print(f"[hf] checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    mode = resolve_mode(args.mode, cfg)

    enc_state, head_state = _split_state(ckpt["model_state"])
    print(f"[hf] encoder={cfg.get('encoder')} head={cfg.get('head', 'auto')} "
          f"frozen={bool(cfg.get('freeze_encoder'))} step={ckpt.get('step')}")
    print(f"[hf] mode={mode}"
          + (f" (encoder {_fmt_bytes(_state_bytes(enc_state))} not uploaded; "
             f"head is {_fmt_bytes(_state_bytes(head_state))})"
             if mode == "head-only" else ""))
    if "optim_state" in ckpt and not args.with_optimizer:
        print("[hf] dropping optimizer state (use --with-optimizer to keep it)")

    metrics = find_metrics(run_dir, tag)
    if metrics:
        print(f"[hf] metrics found for protocol(s): {', '.join(sorted(metrics))}")

    staging = Path(tempfile.mkdtemp(prefix="kgfm-hf-"))
    try:
        sizes = _stage(staging, ckpt, cfg, mode, args.with_optimizer,
                       metrics, args.repo)
        total = sum(sizes.values())
        for name, size in sizes.items():
            print(f"[hf]   {name:20} {_fmt_bytes(size):>10}")
        print(f"[hf] total {_fmt_bytes(total)} -> {args.repo} "
              f"({'private' if args.private else 'PUBLIC'})")

        if args.dry_run:
            keep = Path(args.dry_run_dir) if args.dry_run_dir else None
            if keep:
                keep.mkdir(parents=True, exist_ok=True)
                for p in staging.iterdir():
                    shutil.copy2(p, keep / p.name)
                print(f"[hf] dry run: files written to {keep}, nothing uploaded")
            else:
                print("[hf] dry run: nothing uploaded "
                      "(pass --dry-run-dir to keep the staged files)")
            return

        try:
            from huggingface_hub import HfApi
        except ImportError as exc:            # pragma: no cover
            raise SystemExit(
                "kgfm hf needs huggingface_hub: pip install huggingface_hub"
            ) from exc

        api = HfApi(token=args.token or os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private,
                        exist_ok=True)
        api.upload_folder(
            repo_id=args.repo, folder_path=str(staging), repo_type="model",
            commit_message=args.message,
        )
        print(f"[hf] pushed https://huggingface.co/{args.repo}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load(repo_or_path: str, device: Optional[str] = None, **kwargs: Any):
    """Rebuild a scorer from a Hub repo (or a local uploaded directory).

    Mirrors what `kgfm hf` wrote: for a head-only repo the encoder is
    reconstructed from its preset — which pulls the public weights the frozen
    run used — and the head is loaded over it.
    """
    from .heads import DEFAULT_HEAD
    from .model import DistMultScorer, make_encoder

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    local = Path(repo_or_path)
    if local.is_dir():
        model_path = local / MODEL_FILE
    elif local.is_file():
        model_path = local
    else:
        from huggingface_hub import hf_hub_download

        model_path = Path(hf_hub_download(repo_or_path, MODEL_FILE, **kwargs))

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {}) or {}
    encoder = make_encoder(
        cfg.get("encoder", "ngram"),
        vocab_size=cfg.get("vocab_size", 1 << 20),
        embedding_dim=cfg.get("embedding_dim", 256),
        n_min=cfg.get("n_min", 3),
        n_max=cfg.get("n_max", 5),
        max_ngrams=cfg.get("max_ngrams", 96),
        transformer_model=cfg.get("transformer_model",
                                  "bert-base-multilingual-cased"),
        transformer_max_length=cfg.get("transformer_max_length", 128),
        transformer_pooling=cfg.get("transformer_pooling", "mean"),
        freeze_encoder=cfg.get("freeze_encoder", False),
        encoder_dropout=cfg.get("encoder_dropout"),
    )
    scorer = DistMultScorer(
        encoder, proj_dim=cfg.get("proj_dim"), normalize=True,
        head_dropout=cfg.get("head_dropout", 0.0),
        head=cfg.get("head", DEFAULT_HEAD),
    ).to(dev)
    # strict=False is correct only for head-only payloads, where the encoder
    # keys are intentionally absent; for a full payload nothing is missing.
    head_only = payload.get("kgfm_upload_mode") == "head-only"
    result = scorer.load_state_dict(payload["model_state"], strict=not head_only)
    if head_only:
        # The encoder keys are meant to be missing; anything else is not.
        stray = [k for k in result.missing_keys if not k.startswith("encoder.")]
        if stray or result.unexpected_keys:
            raise SystemExit(
                f"head-only payload does not fit this architecture: "
                f"missing {stray[:5]}, unexpected {list(result.unexpected_keys)[:5]}"
            )
    scorer.eval()
    return scorer


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", required=True, metavar="USER/NAME",
                   help="Target HuggingFace repo id.")
    src = p.add_argument_group("which checkpoint")
    src.add_argument("--ckpt", default=None,
                     help="Checkpoint file to publish.")
    src.add_argument("--out-dir", default=None,
                     help="Benchmark run to take it from ('latest', a "
                          "timestamp, or a path). Use with --tag.")
    src.add_argument("--results-root", default="benchmarks/results/chembl",
                     help="Where run directories live.")
    src.add_argument("--tag", default=None,
                     help="Cell tag inside the run, e.g. bge-large_mlp_frozen.")
    src.add_argument("--which", default="best", choices=list(CKPT_CHOICES),
                     help="Which snapshot of that cell (default: best).")

    up = p.add_argument_group("what to upload")
    up.add_argument("--mode", default="auto", choices=list(UPLOAD_MODES),
                    help="'head-only' skips a frozen encoder's weights, which "
                         "are still the public ones (a frozen bge-large cell "
                         "is 1 MB of head against 1.3 GB of encoder). 'auto' "
                         "picks it whenever that is lossless.")
    up.add_argument("--with-optimizer", action="store_true",
                    help="Keep the optimizer state, roughly doubling the "
                         "upload. Only needed to resume training from the Hub.")

    pub = p.add_argument_group("publishing")
    pub.add_argument("--public", dest="private", action="store_false",
                     default=True,
                     help="Create the repo public. The default is private — "
                          "publishing is hard to undo, so it is opt-in.")
    pub.add_argument("--token", default=None,
                     help="HF token. Falls back to $HF_TOKEN, "
                          "$HUGGINGFACE_HUB_TOKEN, then the cached login.")
    pub.add_argument("--message", default="Upload kgfm scorer",
                     help="Commit message.")
    pub.add_argument("--dry-run", action="store_true",
                     help="Build the upload and report it without touching "
                          "the network.")
    pub.add_argument("--dry-run-dir", default=None,
                     help="With --dry-run, keep the staged files here so you "
                          "can inspect exactly what would be pushed.")
