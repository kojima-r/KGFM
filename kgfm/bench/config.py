"""Benchmark configuration, loaded from YAML.

Scale settings live in ``benchmarks/config_*.yaml`` and are selected with
``--config``. Keeping them as data rather than code means a new scale is a
file you can diff and version, not a dict inside the package.

The file has two levels, because a sweep setting is one of two different
things:

* **run-level** keys sit at the top of the file and describe the run as a
  whole — which data, which cells exist, how many GPUs. They cannot vary by
  cell.
* **cell-level** keys describe *one training pass*. They go under
  ``defaults:`` to apply to every cell, and under ``cells: <tag>:`` to
  override that for one cell. A tag is ``<encoder>`` or ``<encoder>_frozen``
  (see ``BenchConfig.cell_specs``).

::

    prep_max_train: 250000     # run-level
    encoders: [ngram, transformer]

    defaults:                  # every cell
      max_steps: 25000
      batch_size: 512

    cells:                     # one cell
      transformer:
        batch_size: 64

Precedence is dataclass defaults <- ``defaults:`` <- ``cells: <tag>:`` <-
explicit CLI flags. The CLI comes last on purpose: every bench flag defaults
to ``None`` and is a *single global override*, so passing one deliberately
flattens the per-cell settings for every cell. That is the distinction the
two levels exist to make — put anything that should differ per cell in the
file, not on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..encoders import ENCODER_PRESETS, is_frozen_only, is_transformer
from ..heads import DEFAULT_HEAD, HEADS, is_trainable

STEPS = ("prep", "sweep", "viz")

# Settings that describe one training pass and may therefore differ per cell.
# Everything not listed here is run-level and may only appear at the top of a
# config file. Keep in sync with the `_add_*` groups in bench/cli.py.
CELL_FIELDS = frozenset({
    "max_steps", "max_epoch", "batch_size", "proj_dim",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "log_every", "eval_every", "valid_loss_batches",
    "lr", "loss", "loss_temperature",
    "weight_decay", "encoder_weight_decay", "head_weight_decay",
    "encoder_dropout", "head_dropout", "mask_duplicate_tails",
    "n_eval_triples", "pool_size", "max_filter_tails", "max_filter_rows",
    "max_rows_per_file", "ckpt_every",
    "interleave_files", "valid_loss_shuffle",
})
# Renamed settings, kept only to turn a stale config into a clear error.
_RENAMED = {
    "max_train": "prep_max_train",
    "max_valid": "prep_max_valid",
    "max_test": "prep_max_test",
}

PROTOCOLS = ("pooled", "filtered")
FREEZES = ("off", "on")


@dataclass
class BenchConfig:
    """Everything a kgfm ChEMBL benchmark run needs.

    These defaults are the smoke scale; `benchmarks/*.yaml` override them.

    Scope is kgfm only. ULTRA / MOTIF are separate methods with their own
    commands (`kgfm-ultra`, `kgfm-motif`) and their own flags; the only thing
    shared with them is the KG this config builds and the run directory both
    write into.
    """

    # --- environment / layout (paths are relative to the working directory,
    # which the shell wrappers set to the repo root) ---
    conda_env: str = "kgfm"
    results_root: str = "benchmarks/results/chembl"
    # Recorded in meta.json so a run says which settings file produced it.
    config_file: str = ""
    kg_dir: str = "benchmarks/chembl_kg"
    run_label: str = ""

    # --- data ---
    train_list: str = "list_chembl/train.txt"
    valid_list: str = "list_chembl/valid.txt"
    test_list: str = "list_chembl/test.txt"
    # Row caps for `kgfm bench prep`, which turns the raw TSVs into the
    # entity-ID KG that ULTRA and MOTIF consume. They do **not** limit kgfm's
    # own training or evaluation: kgfm streams the TSVs in the lists above
    # directly and never reads the prepared KG. The `prep_` prefix is there so
    # that is obvious from the name — an earlier `max_train` read like a cap on
    # the whole run. Both baselines read the *same* prepared KG, which is why
    # there is one set of caps rather than one per baseline.
    prep_max_train: int = 50_000
    prep_max_valid: int = 2_000
    prep_max_test: int = 2_000
    strict_transductive: bool = False
    seed: int = 0

    # --- kgfm sweep ---
    encoders: List[str] = field(default_factory=lambda: ["ngram", "transformer"])
    # Projection heads to sweep. With one entry the cell tags are unchanged;
    # with more than one, each tag gains a `_<head>` segment so the cells stay
    # distinguishable in the run directory and the report.
    heads: List[str] = field(default_factory=lambda: [DEFAULT_HEAD])
    freezes: List[str] = field(default_factory=lambda: ["off"])
    protocols: List[str] = field(default_factory=lambda: ["pooled", "filtered"])
    # --- cell-level defaults (see CELL_FIELDS). `defaults:` in a config file
    # sets these; `cells: <tag>:` overrides them for one cell. A transformer
    # cell often wants a smaller batch than an ngram one, because
    # encode_triple bundles h+r+t into a single 3B-sequence encoder forward —
    # that is now `cells: transformer: batch_size:`, not a special field.
    max_steps: int = 200
    # Passes over the train data instead of a step count (fractional allowed).
    # Resolved to steps here rather than in the cell, because the derived
    # eval/log cadences below are computed from max_steps and the cell only
    # ever receives the resolved number.
    max_epoch: Optional[float] = None
    batch_size: int = 256
    # Required whenever `freezes` contains "on": with a frozen encoder and no
    # projection the model has zero trainable parameters.
    proj_dim: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    nproc: int = 1
    master_port: int = 29500
    log_every: int = 50
    # In-loop validation cadence. None = derive from max_steps so that every
    # run, however short, produces a validation curve — kgfm.train's own
    # default of 1000 silently skips validation entirely on a 200-step cell.
    eval_every: Optional[int] = None
    valid_loss_batches: int = 10
    # None = kgfm.train picks per encoder (1e-3 ngram / 3e-5 transformer).
    lr: Optional[float] = None
    # Training objective; see kgfm/losses.py. None = kgfm.train's default.
    loss: Optional[str] = None
    loss_temperature: Optional[float] = None
    # Regularization, split between the encoder and the projection head
    # because they overfit at different rates. None = kgfm.train's default.
    weight_decay: Optional[float] = None
    encoder_weight_decay: Optional[float] = None
    head_weight_decay: Optional[float] = None
    encoder_dropout: Optional[float] = None
    head_dropout: Optional[float] = None
    # False negatives from repeated tails; on by default in kgfm.train.
    mask_duplicate_tails: Optional[bool] = None
    # Rows to take from each TSV before moving to the next. None = read the
    # file to the end, which on ChEMBL means never leaving it: every file is
    # 10,000,000 rows, workers read sequentially, and a 25k-step run at B=512
    # only pulls 3.2M rows per worker. So the default touches ~4 files of 85
    # (one per worker, 32% in). Setting this trades depth for breadth.
    max_rows_per_file: Optional[int] = None
    # Steps between `last.pt` rewrites. None = kgfm.train's 1000, which is far
    # too often for a million-step run (each rewrite is the whole model plus
    # optimizer state).
    ckpt_every: Optional[int] = None
    # Data ordering and validation-loss measurement; see kgfm/data.py and
    # train.make_loader. Both default to the corrected behaviour in
    # kgfm.train; set them False only to reproduce a pre-2026-08-25 run.
    interleave_files: Optional[bool] = None
    valid_loss_shuffle: Optional[bool] = None
    n_eval_triples: int = 5_000
    pool_size: int = 5_000
    max_filter_tails: int = 50_000
    max_filter_rows: int = 1_000_000

    # --- embedding projection (`kgfm viz`) ---
    viz_reducer: str = "auto"          # auto -> umap when installed, else pca
    viz_max_points: int = 4_000        # a few thousand keeps the report light

    # --- per-cell overrides, keyed by cell tag (`<encoder>[_frozen]`) ---
    cells: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Flags the user actually typed. Applied *after* the per-cell overrides,
    # because a CLI flag is a single global override by definition.
    cli_overrides: Dict[str, Any] = field(default_factory=dict)
    # Memoized epoch sizes, keyed by the caps that change them. Not a setting.
    _epoch_cache: Dict[Any, int] = field(default_factory=dict, repr=False)

    # --- control ---
    skip: List[str] = field(default_factory=list)
    resume: Optional[str] = None         # None = fresh run; else a run target

    def cell_specs(self) -> List[tuple]:
        """(encoder, head, freeze, tag) for every cell, in sweep order.

        `freeze=on` is only a cell for pretrained encoders — there is no LM to
        freeze in the ngram one, so it would duplicate its `off` cell.
        """
        specs: List[tuple] = []
        multi_head = len(self.heads) > 1
        for encoder in self.encoders:
            for head in self.heads:
                for freeze in self.freezes:
                    if freeze == "on" and not is_transformer(encoder):
                        continue
                    # 7B-class presets cannot be fine-tuned here at all (3B
                    # sequences per step), so their freeze=off cell is not a
                    # cell — the same kind of rule as ngram x freeze=on.
                    if freeze == "off" and is_frozen_only(encoder):
                        continue
                    # The head segment only appears when it distinguishes
                    # something, so single-head configs keep the tags (and
                    # therefore the result filenames) they have always had.
                    tag = encoder + (f"_{head}" if multi_head else "")
                    if freeze == "on":
                        tag += "_frozen"
                    specs.append((encoder, head, freeze, tag))
        return specs

    def cell_tags(self) -> List[str]:
        """Just the tags — the identity of each cell in the run directory."""
        return [spec[3] for spec in self.cell_specs()]

    def _epoch_steps(self, epochs: float, batch_size: int,
                     max_rows_per_file: Optional[int],
                     accum: int) -> int:
        """Steps for `epochs` passes over train_list, memoized per cell shape.

        The row count itself is cached on disk by `data.count_rows`, so this is
        a one-off ~1 minute cost per corpus and instant afterwards.
        """
        import math

        from ..data import epoch_examples, read_file_list

        if epochs <= 0:
            raise SystemExit(f"max_epoch must be positive (got {epochs})")
        key = (max_rows_per_file,)
        if key not in self._epoch_cache:
            files = read_file_list(self.train_list)
            if not files:
                raise SystemExit(
                    f"max_epoch needs a readable train list; {self.train_list} "
                    "listed no files."
                )
            self._epoch_cache[key] = epoch_examples(
                files, max_rows_per_file=max_rows_per_file
            )
        per_step = batch_size * max(1, int(self.nproc)) * max(1, accum)
        return max(1, math.ceil(epochs * self._epoch_cache[key] / per_step))

    def resolve_cell(self, tag: str) -> Dict[str, Any]:
        """Settings for one cell: defaults <- cells[tag] <- CLI overrides.

        The CLI comes last because a bench flag is a single global override;
        anything meant to differ per cell belongs in the file.
        """
        resolved = {name: getattr(self, name) for name in CELL_FIELDS}
        resolved.update(self.cells.get(tag, {}))
        resolved.update({k: v for k, v in self.cli_overrides.items()
                         if k in CELL_FIELDS})

        # `max_epoch` wins over `max_steps`, and is turned into steps *here* so
        # that the cadences below and the `--max-steps` the cell receives all
        # agree on one number. Dividing by nproc is what makes a 1-epoch config
        # correct on any GPU count: each rank reads files[rank::world_size].
        if resolved["max_epoch"] is not None:
            resolved["max_steps"] = self._epoch_steps(
                float(resolved["max_epoch"]),
                int(resolved["batch_size"]),
                resolved["max_rows_per_file"],
                int(resolved["gradient_accumulation_steps"]),
            )

        # Derived cadences, computed from *this cell's* max_steps: a cell may
        # train for a different number of steps than its neighbours, and
        # kgfm.train's own eval_every default of 1000 would skip validation
        # entirely on a short one.
        steps = int(resolved["max_steps"])
        if resolved["eval_every"] is None:
            resolved["eval_every"] = max(1, steps // 10)
        else:
            resolved["eval_every"] = max(1, int(resolved["eval_every"]))
        resolved["log_every"] = max(1, min(int(resolved["log_every"]),
                                           max(1, steps // 20)))
        return resolved

    def validate(self) -> None:
        for name, values, allowed in (
            ("protocols", self.protocols, PROTOCOLS),
            ("freezes", self.freezes, FREEZES),
        ):
            bad = [v for v in values if v not in allowed]
            if bad:
                raise SystemExit(
                    f"Invalid --{name} value(s): {', '.join(bad)} "
                    f"(use {'|'.join(allowed)})"
                )
        bad_steps = [s for s in self.skip if s not in STEPS]
        if bad_steps:
            raise SystemExit(
                f"Invalid --skip value(s): {', '.join(bad_steps)} "
                f"(use {'|'.join(STEPS)})"
            )
        # A typo in a cell tag would otherwise be silently ignored — the cell
        # would just run with the defaults and the override would vanish.
        known = self.cell_tags()
        unknown = [t for t in self.cells if t not in known]
        if unknown:
            raise SystemExit(
                f"Unknown cell tag(s) in `cells:`: {', '.join(sorted(unknown))}\n"
                f"This run's cells are: {', '.join(known) or '(none)'}\n"
                "A tag is <encoder>[_<head>][_frozen] — the _<head> segment "
                "appears only when more than one head is swept — and must be "
                "a cell that `encoders` x `heads` x `freezes` produces."
            )
        bad_heads = [h for h in self.heads if h not in HEADS]
        if bad_heads:
            raise SystemExit(
                f"Invalid heads value(s): {', '.join(bad_heads)} "
                f"(use {'|'.join(HEADS)})"
            )
        for encoder, head, freeze, tag in self.cell_specs():
            if freeze != "on":
                continue
            cell = self.resolve_cell(tag)
            # A frozen encoder trains only the head, so a head with no
            # parameters means the cell trains nothing at all.
            in_dim = ENCODER_PRESETS.get(encoder, {}).get("dim")
            if not is_trainable(head, in_dim or 0, cell["proj_dim"]):
                print(
                    f"[bench] warning: cell {tag} is frozen and head={head} has "
                    "no parameters; it would train nothing. Set proj_dim, or "
                    "use head=linear/mlp/residual_mlp."
                )

    def as_meta(self) -> Dict[str, Any]:
        """The parameter block recorded in meta.json."""
        skip_fields = {"skip", "resume", "_epoch_cache"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in skip_fields
        }


# ---------------------------------------------------------------------------
# YAML config files
# ---------------------------------------------------------------------------

# Searched in order for a bare `--config <name>`. `benchmark_scaling/` mirrors
# `benchmarks/` and holds the scaling-study configs; they are the same file
# format and run through the same pipeline, so they share the lookup.
CONFIG_DIRS = ("benchmarks", "benchmark_scaling")
CONFIG_DIR = CONFIG_DIRS[0]          # default home for a new config
# The prefix keeps settings files distinguishable from any other YAML that
# ends up in benchmarks/ (upstream clones bring their own configs).
CONFIG_GLOB = "config_*.yaml"

# YAML 1.1 booleans: a bare `off` / `on` in a list parses as False / True, so
# `freezes: [off, on]` would silently become `[False, True]` and then fail
# validation with a confusing message. Map them back for the fields that take
# these words. Quoting in the file also works; this makes both spellings safe.
_BOOL_WORDS = {True: "on", False: "off"}


def _coerce(field: str, value: Any) -> Any:
    if field == "freezes" and isinstance(value, list):
        return [_BOOL_WORDS.get(v, v) if isinstance(v, bool) else v for v in value]
    return value


def load_config_file(path: str) -> Dict[str, Any]:
    """Read a benchmark settings file into a plain dict of overrides."""
    try:
        import yaml
    except ImportError as exc:                              # pragma: no cover
        raise SystemExit(
            "Reading --config needs PyYAML: pip install pyyaml"
        ) from exc

    file_path = Path(path)
    if not file_path.is_file():
        # A bare name is looked up in benchmarks/, with and without the
        # `config_` prefix, so both `--config large` and `--config config_large`
        # find benchmarks/config_large.yaml.
        for directory in CONFIG_DIRS:
            for name in (f"{path}.yaml", f"config_{path}.yaml"):
                candidate = Path(directory) / name
                if candidate.is_file():
                    file_path = candidate
                    break
            if file_path.is_file():
                break
        else:
            raise SystemExit(
                f"Config file not found: {path}\n"
                f"Available: {', '.join(available_configs()) or '(none)'}"
            )

    loaded = yaml.safe_load(file_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{file_path}: expected a mapping at the top level")

    return _parse_config(loaded, file_path)


def _check_cell_block(where: str, block: Any, file_path: Path) -> Dict[str, Any]:
    """Validate one `defaults:` / `cells: <tag>:` mapping."""
    if not isinstance(block, dict):
        raise SystemExit(f"{file_path}: `{where}` must be a mapping")
    unknown = sorted(set(block) - CELL_FIELDS)
    if unknown:
        run_level = [k for k in unknown
                     if k in {f.name for f in fields(BenchConfig)}]
        hint = ""
        if run_level:
            plural = "s" if len(run_level) > 1 else ""
            hint = (f"\n{', '.join(run_level)} {'are' if plural else 'is'} a "
                    f"run-level setting{plural} — move "
                    f"{'them' if plural else 'it'} to the top level of the file.")
        raise SystemExit(
            f"{file_path}: unknown setting(s) under `{where}`: "
            f"{', '.join(unknown)}{hint}\n"
            f"Valid cell settings: {', '.join(sorted(CELL_FIELDS))}"
        )
    return dict(block)


def _parse_config(loaded: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    """Flatten a two-level config file into BenchConfig field overrides."""
    all_fields = {f.name for f in fields(BenchConfig)}
    # `cli_overrides` is populated from argparse, never from a file.
    top_level = all_fields - CELL_FIELDS - {"cli_overrides", "_epoch_cache"}

    overrides: Dict[str, Any] = {}
    cells: Dict[str, Dict[str, Any]] = {}
    for key, value in loaded.items():
        if key == "defaults":
            # `defaults:` sets the cell-level fields on the dataclass itself,
            # which is exactly what resolve_cell() starts from.
            overrides.update(_check_cell_block("defaults", value, file_path))
        elif key == "cells":
            if not isinstance(value, dict):
                raise SystemExit(f"{file_path}: `cells` must be a mapping of "
                                 "cell tag -> settings")
            for tag, block in value.items():
                cells[str(tag)] = _check_cell_block(
                    f"cells.{tag}", block, file_path
                )
        elif key in top_level:
            overrides[key] = _coerce(key, value)
        elif key in CELL_FIELDS:
            raise SystemExit(
                f"{file_path}: `{key}` is a cell-level setting and cannot sit "
                f"at the top level.\nPut it under `defaults:` to apply it to "
                f"every cell, or under `cells: <tag>:` for one cell."
            )
        elif key in _RENAMED:
            raise SystemExit(
                f"{file_path}: `{key}` was renamed to `{_RENAMED[key]}`.\n"
                "It only caps `kgfm bench prep` (the entity-ID KG that ULTRA "
                "and MOTIF read) and never limited kgfm's own training, which "
                "the old name obscured."
            )
        else:
            raise SystemExit(
                f"{file_path}: unknown setting: {key}\n"
                f"Run-level settings: {', '.join(sorted(top_level))}\n"
                f"Cell-level settings (under `defaults:` / `cells:`): "
                f"{', '.join(sorted(CELL_FIELDS))}"
            )

    if cells:
        overrides["cells"] = cells
    overrides["config_file"] = str(file_path)
    return overrides


def available_configs() -> List[str]:
    """YAML settings files shipped under any of ``CONFIG_DIRS``."""
    found: List[str] = []
    for directory in CONFIG_DIRS:
        d = Path(directory)
        if d.is_dir():
            found.extend(sorted(p.stem for p in d.glob(CONFIG_GLOB)))
    return found


def build_config(
    overrides: Dict[str, Any], config_path: Optional[str] = None
) -> BenchConfig:
    """Compose dataclass defaults <- config file <- explicit CLI overrides.

    CLI values are also kept in `cfg.cli_overrides` so `resolve_cell` can
    re-apply them *after* the per-cell block: a bench flag is a single global
    override and must win over a per-cell setting, not lose to it.
    """
    cfg = BenchConfig()
    if config_path:
        for key, value in load_config_file(config_path).items():
            setattr(cfg, key, value)
    typed = {k: v for k, v in overrides.items()
             if v is not None and hasattr(cfg, k)}
    for key, value in typed.items():
        setattr(cfg, key, value)
    cfg.cli_overrides = {k: v for k, v in typed.items() if k in CELL_FIELDS}
    return cfg
