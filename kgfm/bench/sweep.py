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


def run_step(cfg: BenchConfig, out_dir: Path, logger: RunLogger) -> List[Path]:
    """Run every cell, returning the JSON records that now exist."""
    produced: List[Path] = []
    kgfm = envs.kgfm_cmd(cfg.conda_env)
    torchrun = envs.torchrun_cmd(cfg.conda_env)

    logger.log(
        f"kgfm sweep: encoders={cfg.encoders} heads={cfg.heads} "
        f"freezes={cfg.freezes} protocols={cfg.protocols}"
    )
    logger.log(f"    python={kgfm[0]} nproc={cfg.nproc}")

    # cell_specs() owns which (encoder, head, freeze) combinations are real
    # cells and what each is called, so the sweep and the config agree on the
    # cell list by construction rather than by two copies of the same rule.
    for encoder, head, freeze, tag in cfg.cell_specs():
        ckpt_dir = out_dir / f"kgfm_ckpts_{tag}"
        # defaults: <- cells: <tag>: <- CLI flags, resolved once per cell.
        cell = cfg.resolve_cell(tag)
        if cfg.cells.get(tag):
            logger.log(f"cell {tag}: overrides "
                       + " ".join(f"{k}={v}" for k, v in
                                  sorted(cfg.cells[tag].items())))

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
                "--head", head,
                "--max-steps", cell["max_steps"],
                "--batch-size", cell["batch_size"],
                "--protocol", protocol,
                "--n-eval-triples", cell["n_eval_triples"],
                "--pool-size", cell["pool_size"],
                "--max-filter-tails", cell["max_filter_tails"],
                "--max-filter-rows", cell["max_filter_rows"],
                "--train-list", cfg.train_list,
                "--valid-list", cfg.valid_list,
                "--test-list", cfg.test_list,
                "--gradient-accumulation-steps",
                cell["gradient_accumulation_steps"],
                "--log-every", cell["log_every"],
                "--eval-every", cell["eval_every"],
                "--valid-loss-batches", cell["valid_loss_batches"],
                "--seed", cfg.seed,
                "--ckpt-dir", ckpt_dir,
                "--out", out_path,
            ]
            if freeze == "on":
                cell_args.append("--freeze-encoder")
            for flag, key in (
                ("--lr", "lr"),
                ("--loss", "loss"),
                ("--loss-temperature", "loss_temperature"),
                ("--weight-decay", "weight_decay"),
                ("--encoder-weight-decay", "encoder_weight_decay"),
                ("--head-weight-decay", "head_weight_decay"),
                ("--encoder-dropout", "encoder_dropout"),
                ("--head-dropout", "head_dropout"),
                ("--proj-dim", "proj_dim"),
                ("--max-rows-per-file", "max_rows_per_file"),
                ("--per-device-train-batch-size",
                 "per_device_train_batch_size"),
                ("--per-device-eval-batch-size",
                 "per_device_eval_batch_size"),
            ):
                if cell[key] is not None:
                    cell_args += [flag, cell[key]]
            if cell["mask_duplicate_tails"] is False:
                cell_args.append("--no-mask-duplicate-tails")

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
