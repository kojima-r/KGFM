"""Benchmark configuration, loaded from YAML.

Scale settings live in ``benchmarks/config_*.yaml`` and are selected with
``--config``. Keeping them as data rather than code means a new scale is a
file you can diff and version, not a dict inside the package.

Precedence is defaults <- config file <- explicit CLI flags: every CLI flag
defaults to ``None``, the file fills the Nones, and anything the user actually
typed wins over both.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

STEPS = ("prep", "sweep", "viz")
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

    # --- data caps for the prepared entity-ID KG ---
    train_list: str = "list_chembl/train.txt"
    valid_list: str = "list_chembl/valid.txt"
    test_list: str = "list_chembl/test.txt"
    max_train: int = 50_000
    max_valid: int = 2_000
    max_test: int = 2_000
    strict_transductive: bool = False
    seed: int = 0

    # --- kgfm sweep ---
    encoders: List[str] = field(default_factory=lambda: ["ngram", "transformer"])
    freezes: List[str] = field(default_factory=lambda: ["off"])
    protocols: List[str] = field(default_factory=lambda: ["pooled", "filtered"])
    max_steps: int = 200
    batch_size: int = 256
    # Transformer cells often need a smaller batch than ngram ones: encode_triple
    # bundles h+r+t into a single 3B-sequence encoder forward, so B=1024 is a
    # 3072-sequence BERT batch. None = use batch_size everywhere.
    transformer_batch_size: Optional[int] = None
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
    n_eval_triples: int = 5_000
    pool_size: int = 5_000
    max_filter_tails: int = 50_000
    max_filter_rows: int = 1_000_000

    # --- embedding projection (`kgfm viz`) ---
    viz_reducer: str = "auto"          # auto -> umap when installed, else pca
    viz_max_points: int = 4_000        # a few thousand keeps the report light

    # --- control ---
    skip: List[str] = field(default_factory=list)
    resume: Optional[str] = None         # None = fresh run; else a run target

    def resolved_eval_every(self) -> int:
        """Validation cadence: explicit if given, else ~10 points per run."""
        if self.eval_every is not None:
            return max(1, self.eval_every)
        return max(1, self.max_steps // 10)

    def resolved_log_every(self) -> int:
        """Loss-logging cadence, capped so short runs still get a curve."""
        return max(1, min(self.log_every, max(1, self.max_steps // 20)))

    def transformer_bs(self) -> int:
        return self.transformer_batch_size or self.batch_size

    def cell_batch_size(self, encoder: str) -> int:
        return self.batch_size if encoder == "ngram" else self.transformer_bs()

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
        if "on" in self.freezes and self.proj_dim is None:
            # Not fatal — the user may be probing — but it silently trains
            # nothing, which is worth one line of warning.
            print(
                "[bench] warning: --freezes contains 'on' without --proj-dim; "
                "frozen cells will have zero trainable parameters."
            )

    def as_meta(self) -> Dict[str, Any]:
        """The parameter block recorded in meta.json."""
        skip_fields = {"skip", "resume"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in skip_fields
        }


# ---------------------------------------------------------------------------
# YAML config files
# ---------------------------------------------------------------------------

CONFIG_DIR = "benchmarks"
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
        for name in (f"{path}.yaml", f"config_{path}.yaml"):
            candidate = Path(CONFIG_DIR) / name
            if candidate.is_file():
                file_path = candidate
                break
        else:
            raise SystemExit(
                f"Config file not found: {path}\n"
                f"Available: {', '.join(available_configs()) or '(none)'}"
            )

    loaded = yaml.safe_load(file_path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"{file_path}: expected a mapping at the top level")

    known = {f.name for f in fields(BenchConfig)}
    unknown = sorted(set(loaded) - known)
    if unknown:
        raise SystemExit(
            f"{file_path}: unknown setting(s): {', '.join(unknown)}\n"
            f"Valid settings: {', '.join(sorted(known))}"
        )
    overrides = {k: _coerce(k, v) for k, v in loaded.items()}
    overrides["config_file"] = str(file_path)
    return overrides


def available_configs() -> List[str]:
    """YAML settings files shipped under ``benchmarks/``."""
    directory = Path(CONFIG_DIR)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob(CONFIG_GLOB))


def build_config(
    overrides: Dict[str, Any], config_path: Optional[str] = None
) -> BenchConfig:
    """Compose defaults <- config file <- explicit CLI overrides."""
    cfg = BenchConfig()
    if config_path:
        for key, value in load_config_file(config_path).items():
            setattr(cfg, key, value)
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
