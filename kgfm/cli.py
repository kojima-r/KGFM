"""The `kgfm` command.

    kgfm train ...      train a DistMult-over-text-encoder scorer
    kgfm eval  ...      score a checkpoint (MRR / Hit@k / nDCG)
    kgfm bench ...      kgfm's ChEMBL benchmark (prep + sweep)
    kgfm viz ...        project a checkpoint's h / t embeddings to 2D
    kgfm report ...     collect a run directory's results into one table
    kgfm hf    ...      publish a trained scorer to the HuggingFace Hub
    kgfm scaling ...    scaling-law plots (compute vs loss) for a run

The baselines it is compared against are separate methods with separate
commands — `kgfm-ultra` and `kgfm-motif` (see `kgfm.baselines`). They share
only the run directory, which is what `kgfm report` reads.

`python -m kgfm ...` is the same entry point, which is what the benchmark
uses when it needs to re-enter this CLI in a specific conda env or under
torchrun.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kgfm",
        description="Streaming knowledge-graph foundation model over raw text.",
    )
    p.add_argument("--version", action="version", version=f"kgfm {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    # Import the functions, not the modules: `kgfm/__init__.py` re-exports a
    # `train` function that shadows the `kgfm.train` submodule attribute.
    from . import hf as hf_mod
    from .scaling import report as scaling_mod
    from . import report as report_mod
    from . import viz as viz_mod
    from .bench import cli as bench_cli
    from .eval import add_arguments as add_eval_args, run_from_args as run_eval
    from .train import (
        add_arguments as add_train_args,
        config_from_args,
        train as run_train,
    )

    train_p = sub.add_parser("train", help="train a scorer")
    add_train_args(train_p)
    train_p.set_defaults(func=lambda a: run_train(config_from_args(a)))

    eval_p = sub.add_parser("eval", help="evaluate a checkpoint")
    add_eval_args(eval_p)
    eval_p.set_defaults(func=run_eval)

    bench_cli.build_parser(sub)

    viz_p = sub.add_parser(
        "viz", help="project a checkpoint's h / t embeddings to 2D")
    viz_mod.add_arguments(viz_p)
    viz_p.set_defaults(func=viz_mod.run_from_args)

    report_p = sub.add_parser(
        "report", help="collect a run directory's results into one table")
    report_mod.add_arguments(report_p)
    report_p.set_defaults(func=report_mod.run_from_args)

    hf_p = sub.add_parser(
        "hf", help="publish a trained scorer to the HuggingFace Hub")
    hf_mod.add_arguments(hf_p)
    hf_p.set_defaults(func=hf_mod.run_from_args)

    scaling_p = sub.add_parser(
        "scaling", help="scaling-law plots (compute vs loss) for a run")
    scaling_mod.add_arguments(scaling_p)
    scaling_p.set_defaults(func=scaling_mod.run_from_args)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return
    func(args)


if __name__ == "__main__":
    main()
