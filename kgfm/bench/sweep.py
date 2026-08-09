"""The kgfm sweep: encoders x freezes x protocols.

Each (encoder, freeze) pair is a *cell* and trains exactly once. The first
protocol to run for a cell owns the training pass; later protocols re-score
that same checkpoint with --skip-train, because the protocol only changes
ranking, not the model. `ngram x freeze=on` is skipped — there is no LM to
freeze, so it would duplicate the `ngram x off` cell.

Cells run as child processes (`kgfm bench cell`, optionally under torchrun)
rather than in-process: training sets up CUDA and torch.distributed state that
does not survive being repeated in one interpreter, and multi-GPU cells need a
launcher anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .. import envs
from ..runs import RunLogger
from .config import BenchConfig


def cell_tag(encoder: str, freeze: str) -> str:
    return f"{encoder}_frozen" if freeze == "on" else encoder


def run_step(cfg: BenchConfig, out_dir: Path, logger: RunLogger) -> List[Path]:
    """Run every cell, returning the JSON records that now exist."""
    produced: List[Path] = []
    kgfm = envs.kgfm_cmd(cfg.conda_env)
    torchrun = envs.torchrun_cmd(cfg.conda_env)

    logger.log(
        f"kgfm sweep: encoders={cfg.encoders} freezes={cfg.freezes} "
        f"protocols={cfg.protocols}"
    )
    logger.log(f"    python={kgfm[0]} nproc={cfg.nproc}")

    for encoder in cfg.encoders:
        for freeze in cfg.freezes:
            if encoder == "ngram" and freeze == "on":
                logger.log("skip kgfm cell encoder=ngram freeze=on (no LM to freeze)")
                continue

            tag = cell_tag(encoder, freeze)
            ckpt_dir = out_dir / f"kgfm_ckpts_{tag}"
            batch_size = cfg.cell_batch_size(encoder)

            # Tracks whether this cell's training pass has already been
            # launched in this invocation; later protocols only re-score it.
            trained_yet = False
            for protocol in cfg.protocols:
                out_name = f"kgfm_{protocol}_{tag}.json"
                out_path = out_dir / out_name
                step_name = f"cell_{protocol}_{tag}"

                if cfg.resume and out_path.is_file():
                    logger.log(f"skip {step_name} (resume: {out_name} exists)")
                    produced.append(out_path)
                    continue

                cell_args = [
                    "bench", "cell",
                    "--encoder", encoder,
                    "--max-steps", cfg.max_steps,
                    "--batch-size", batch_size,
                    "--protocol", protocol,
                    "--n-eval-triples", cfg.n_eval_triples,
                    "--pool-size", cfg.pool_size,
                    "--max-filter-tails", cfg.max_filter_tails,
                    "--max-filter-rows", cfg.max_filter_rows,
                    "--train-list", cfg.train_list,
                    "--valid-list", cfg.valid_list,
                    "--test-list", cfg.test_list,
                    "--gradient-accumulation-steps", cfg.gradient_accumulation_steps,
                    "--log-every", cfg.resolved_log_every(),
                    "--eval-every", cfg.resolved_eval_every(),
                    "--valid-loss-batches", cfg.valid_loss_batches,
                    "--seed", cfg.seed,
                    "--ckpt-dir", ckpt_dir,
                    "--out", out_path,
                ]
                if freeze == "on":
                    cell_args.append("--freeze-encoder")
                if cfg.lr is not None:
                    cell_args += ["--lr", cfg.lr]
                if cfg.loss is not None:
                    cell_args += ["--loss", cfg.loss]
                if cfg.loss_temperature is not None:
                    cell_args += ["--loss-temperature", cfg.loss_temperature]
                for flag, value in (
                    ("--weight-decay", cfg.weight_decay),
                    ("--encoder-weight-decay", cfg.encoder_weight_decay),
                    ("--head-weight-decay", cfg.head_weight_decay),
                    ("--encoder-dropout", cfg.encoder_dropout),
                    ("--head-dropout", cfg.head_dropout),
                ):
                    if value is not None:
                        cell_args += [flag, value]
                if cfg.mask_duplicate_tails is False:
                    cell_args.append("--no-mask-duplicate-tails")
                if cfg.proj_dim is not None:
                    cell_args += ["--proj-dim", cfg.proj_dim]
                if cfg.per_device_train_batch_size is not None:
                    cell_args += ["--per-device-train-batch-size",
                                  cfg.per_device_train_batch_size]
                if cfg.per_device_eval_batch_size is not None:
                    cell_args += ["--per-device-eval-batch-size",
                                  cfg.per_device_eval_batch_size]

                if trained_yet:
                    cell_args.append("--skip-train")
                    launcher = list(kgfm)
                else:
                    # First protocol for this cell owns training. Under resume,
                    # the cell picks up final.pt > last.pt > best.pt and
                    # continues from its saved step up to --max-steps.
                    if cfg.resume:
                        cell_args.append("--resume")
                    if cfg.nproc > 1:
                        launcher = torchrun + [
                            "--standalone",
                            f"--nproc-per-node={cfg.nproc}",
                            f"--master-port={cfg.master_port}",
                            "-m", "kgfm",
                        ]
                    else:
                        launcher = list(kgfm)
                    trained_yet = True

                logger.run(launcher + cell_args, step=step_name, tag=tag)
                if out_path.is_file():
                    produced.append(out_path)

    return produced
