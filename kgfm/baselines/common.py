"""Shared machinery for the ULTRA / MOTIF baseline commands.

These are *other people's methods*, not part of kgfm: they consume an
entity-ID KG (built by `kgfm bench prep`), run their own ``script/run.py``
from a released checkpoint with ``--epochs 0``, and drop a JSON record next
to kgfm's results so `kgfm report` can put them in one table. MOTIF is an
ULTRA-derived codebase and shares the flag contract
(``--dataset --gpus --epochs --bpe --ckpt``), so one module covers both.

Upstream patching
-----------------
Neither repo knows about our data, and ULTRA resolves dataset classes via
``getattr(ultra.datasets, <ClassName>)`` — a separate module would not be
found. So we *append* a ``ChEMBLCustom`` class into ``<pkg>/datasets.py``,
guarded by sentinel comments and re-injected on every run so template edits
take effect. Two more patches ride along:

- ``torch.load(self.processed_paths[0])`` gets ``weights_only=False``, because
  PyTorch 2.6 flipped that default and it rejects the upstream pickle caches.
- ``layers.py`` parses ``torch_geometric.__version__`` with a bare ``int()``
  per dot-component, which raises on any PEP 440 suffix (``2.8.0.post1``).
- ``script/run.py`` gets a ``#test_triplets: N`` log line, which is the only
  way to recover how many triples were actually evaluated.

To undo any of it: delete the sentinel-wrapped block and drop the
``weights_only=False``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .. import envs
from ..runs import RunLogger, resolve_run_dir

# What these two import on top of kgfm's own dependencies.
DEPS = ("torch", "torch_geometric", "torch_scatter", "easydict", "yaml")

# Both baselines run in their own conda env, not kgfm's. They JIT-compile an
# `rspmm` CUDA extension against the env's torch, so they need an nvcc that
# matches `torch.version.cuda`; `kgfm` tracks current torch and has no CUDA
# toolchain, and pinning one there would drag kgfm's own torch with it.
# Build this env with `bash benchmarks/setup_baseline_env.sh`.
DEFAULT_ENV = "kgfm-ultra"

_METRIC_RE = re.compile(
    r"(mrr|hits@1|hits@3|hits@10|mr)\s*[:=]\s*([0-9eE+\-.]+)", re.IGNORECASE
)
_COUNT_RE = re.compile(r"#test_triplets:\s*(\d+)")


@dataclass
class Baseline:
    """Everything that differs between ULTRA and MOTIF."""

    method: str                 # "ULTRA" / "MOTIF"
    package: str                # python package inside the clone
    repo_url: str
    default_repo_dir: str
    default_ckpt: str
    default_config: str
    default_gpus: str
    processed_dir: str          # separate caches so the two don't collide
    ckpt_url: Optional[str] = None
    # False when the method cannot run on CPU at all, so `--gpus null` can be
    # rejected up front instead of failing deep inside a forward pass.
    cpu_supported: bool = True
    cpu_unsupported_reason: str = ""

    @property
    def command(self) -> str:
        return f"kgfm-{self.method.lower()}"


# ---------------------------------------------------------------------------
# upstream clone
# ---------------------------------------------------------------------------


def clone_or_update(spec: Baseline, repo_dir: str) -> None:
    dest = Path(repo_dir)
    if (dest / ".git").is_dir():
        print(f"[{spec.command}] {dest} already cloned — pulling latest")
        if subprocess.call(["git", "-C", str(dest), "pull", "--ff-only"]) != 0:
            print(f"[{spec.command}] (warn) pull failed for {dest}")
        return
    print(f"[{spec.command}] cloning {spec.repo_url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.call(["git", "clone", "--depth", "1", spec.repo_url, str(dest)]) != 0:
        print(f"[{spec.command}] (warn) clone failed for {spec.repo_url}")


def fetch_ckpt(spec: Baseline, ckpt_path: str) -> None:
    target = Path(ckpt_path)
    if target.is_file():
        print(f"[{spec.command}] {target} already present")
        return
    if not spec.ckpt_url:
        print(f"[{spec.command}] no download URL known; fetch {target.name} manually")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{spec.command}] fetching {target.name}")
    if subprocess.call(["curl", "-L", "-o", str(target), spec.ckpt_url]) != 0:
        print(f"[{spec.command}] (warn) download failed; fetch manually.")


def check_deps(spec: Baseline, conda_env: str) -> List[str]:
    """Report which imports are unavailable, with the exact fix."""
    python = envs.env_python(conda_env)
    print(f"[{spec.command}] dependency check in conda env '{conda_env}' ({python})")
    missing = envs.missing_modules(DEPS, conda_env)
    if missing:
        print(f"[{spec.command}] missing: {', '.join(missing)}")
        print(f"[{spec.command}] install them with:")
        print(f"[{spec.command}]     ENV_NAME={conda_env} bash benchmarks/setup_baseline_env.sh")
    else:
        print(f"[{spec.command}] all imports available")
    return missing


# ---------------------------------------------------------------------------
# upstream patching
# ---------------------------------------------------------------------------


def _shim_source(marker: str, end: str, data_root_abs: str,
                 processed_dir: str, regen_cmd: str) -> str:
    return f'''
{marker}
# Auto-generated; safe to delete. Re-run `{regen_cmd}` to recreate.
import os as _os_chembl

class ChEMBLCustom(TransductiveDataset):
    name = "chembl_custom"
    # MUST be a literal tab — the prepared TSVs may contain entities whose
    # surface form has internal spaces (e.g. RDF literal strings), so the
    # parent's default ``l.split()`` would mis-tokenize those rows.
    delimiter = "\t"

    _DATA_DIR = {data_root_abs!r}

    def __init__(self, root=None, **kwargs):
        # We accept a 'root' for compatibility but always serve files from
        # _DATA_DIR. Each call still needs *some* root for InMemoryDataset's
        # bookkeeping (the processed cache lives there).
        if root is None:
            root = self._DATA_DIR
        super().__init__(root=root, **kwargs)

    @property
    def raw_dir(self):
        return self._DATA_DIR

    @property
    def processed_dir(self):
        return _os_chembl.path.join(self._DATA_DIR, {processed_dir!r})

    def download(self):
        for fn in self.raw_file_names:
            p = _os_chembl.path.join(self.raw_dir, fn)
            if not _os_chembl.path.exists(p):
                raise FileNotFoundError(
                    f"{{p}} not found. Run `kgfm bench prep` first."
                )
{end}
'''


def _inject_shim(spec: Baseline, repo_dir: str, data_root_abs: str) -> None:
    marker = f"# === BEGIN kgfm injection ({spec.package}: ChEMBLCustom) ==="
    end = f"# === END kgfm injection ({spec.package}: ChEMBLCustom) ==="
    path = os.path.join(repo_dir, spec.package, "datasets.py")
    with open(path, "r") as f:
        content = f.read()

    # Two rewrites of the processed-cache load, applied in sequence so a file
    # patched by an older version of this code gets upgraded rather than
    # skipped:
    #   weights_only=False   — PyTorch 2.6 flipped the default and it rejects
    #                          the upstream pickles.
    #   map_location="cpu"   — the cache keeps whatever device its tensors had
    #                          when it was written, so a cache built while a
    #                          GPU was visible cannot be read back on a CPU-only
    #                          run ("Attempting to deserialize object on a CUDA
    #                          device"). Dataset tensors belong on CPU anyway;
    #                          the model moves what it needs.
    fixed = ('torch.load(self.processed_paths[0], weights_only=False, '
             'map_location="cpu")')
    # Idempotent as plain replacements: neither stale form is a substring of
    # `fixed` (both end in `)` right where `fixed` continues with `,`).
    for stale in (
        "torch.load(self.processed_paths[0])",
        "torch.load(self.processed_paths[0], weights_only=False)",
    ):
        content = content.replace(stale, fixed)

    if marker in content:
        # Strip any prior injection so we can refresh _DATA_DIR.
        start = content.index(marker)
        stop = content.index(end, start) + len(end)
        content = content[:start].rstrip() + "\n" + content[stop:].lstrip()

    if not content.endswith("\n"):
        content += "\n"
    content += _shim_source(
        marker, end, data_root_abs, spec.processed_dir, spec.command
    )

    with open(path, "w") as f:
        f.write(content)


def _patch_pyg_version(spec: Baseline, repo_dir: str) -> None:
    """Make ``layers.py``'s torch_geometric version parse tolerate suffixes.

    Upstream does ``[int(i) for i in torch_geometric.__version__.split(".")]``,
    which dies on a PEP 440 local/post segment — torch_geometric 2.8.0.post1
    raises ``invalid literal for int() with base 10: 'post1'`` deep inside the
    first forward pass. Only the minor component is ever read, so dropping the
    non-numeric parts preserves the behaviour exactly.
    """
    path = os.path.join(repo_dir, spec.package, "layers.py")
    if not os.path.isfile(path):
        return
    needle = '[int(i) for i in torch_geometric.__version__.split(".")]'
    repl = '[int(i) for i in torch_geometric.__version__.split(".") if i.isdigit()]'
    with open(path, "r") as f:
        content = f.read()
    if needle not in content or repl in content:
        return
    with open(path, "w") as f:
        f.write(content.replace(needle, repl))


def _inject_test_count(spec: Baseline, repo_dir: str) -> None:
    """Patch ``script/run.py`` so it logs ``#test_triplets: N`` per split."""
    marker = f"# === BEGIN kgfm injection ({spec.package}: test_count) ==="
    end = f"# === END kgfm injection ({spec.package}: test_count) ==="
    path = os.path.join(repo_dir, "script", "run.py")
    with open(path, "r") as f:
        content = f.read()

    # Strip any prior injection (with its indent and trailing newline).
    while marker in content:
        marker_at = content.index(marker)
        start = content.rfind("\n", 0, marker_at) + 1
        end_marker = content.index(end, marker_at)
        end_line = content.index("\n", end_marker) + 1
        content = content[:start] + content[end_line:]

    needle = (
        "    test_triplets = torch.cat([test_data.target_edge_index,"
        " test_data.target_edge_type.unsqueeze(0)]).t()\n"
    )
    if needle in content:
        insert = (
            f"    {marker}\n"
            f"    if rank == 0:\n"
            f'        logger.warning("#test_triplets: %d" % len(test_triplets))\n'
            f"    {end}\n"
        )
        content = content.replace(needle, needle + insert, 1)

    with open(path, "w") as f:
        f.write(content)


def _cpu_config(spec: Baseline, repo_dir: str, config_rel: str) -> str:
    """Return a config with Triton kernels off, for CPU runs.

    The Triton kernels call ``torch.cuda.set_device`` outright, so any config
    with ``use_triton: yes`` dies on CPU with "Expected a cuda device". The
    flag is a config literal, not one of the ``{{ var }}`` placeholders that
    upstream turns into CLI flags, so the only way to reach the reference path
    is a modified config. Written next to the original inside the (gitignored)
    clone; configs without the flag are returned unchanged.

    Note this is *not* sufficient for MOTIF: ``models.py`` builds its
    HypergraphLayer without forwarding ``use_triton``, so that layer takes the
    Triton path whatever the config says. Hence `cpu_supported=False` there.
    """
    src = os.path.join(repo_dir, config_rel)
    if not os.path.isfile(src):
        return config_rel
    with open(src) as f:
        text = f.read()
    if "use_triton" not in text:
        return config_rel
    patched = re.sub(r"(use_triton\s*:\s*)(yes|true|True|1)\b", r"\1no", text)
    if patched == text:
        return config_rel
    root, ext = os.path.splitext(config_rel)
    dst_rel = f"{root}_kgfm_cpu{ext}"
    with open(os.path.join(repo_dir, dst_rel), "w") as f:
        f.write(patched)
    return dst_rel


def parse_metrics(output: str) -> dict:
    """Pick the *last* occurrence of each metric in the model's output."""
    found: dict = {}
    for m in _METRIC_RE.finditer(output):
        try:
            found[m.group(1).lower()] = float(m.group(2))
        except ValueError:
            pass
    counts = _COUNT_RE.findall(output)
    if counts:
        # Upstream calls test() twice (valid then test); the last line is test.
        found["n"] = int(counts[-1])
    return found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser(spec: Baseline) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=spec.command,
        description=(
            f"Zero-shot {spec.method} inference on a prepared ChEMBL KG. "
            f"Writes <run-dir>/{spec.method.lower()}.json; collect it with "
            f"`kgfm report`."
        ),
    )
    p.add_argument("--out-dir", default="latest",
                   help="Run directory to write into. 'latest', a bare "
                        "timestamp, or a path (default: latest).")
    p.add_argument("--results-root", default="benchmarks/results/chembl",
                   help="Where run directories live.")
    p.add_argument("--kg-dir", default="benchmarks/chembl_kg",
                   help="Entity-ID KG built by `kgfm bench prep`.")
    p.add_argument("--repo-dir", default=spec.default_repo_dir,
                   help=f"Upstream {spec.method} clone.")
    p.add_argument("--ckpt", default=None,
                   help=f"Pretrained checkpoint (default: {spec.default_ckpt}).")
    p.add_argument("--gpus", default=spec.default_gpus,
                   help=f"GPU JSON passed through to {spec.method} "
                        f"(default: {spec.default_gpus}; 'null' means CPU).")
    p.add_argument("--config", default=spec.default_config,
                   help=f"YAML config, relative to --repo-dir.")
    p.add_argument("--bpe", default="null",
                   help="batch_per_epoch; 'null' uses the full data.")
    p.add_argument("--conda-env", default=DEFAULT_ENV,
                   help=f"Conda env to run the upstream script in "
                        f"(default: {DEFAULT_ENV}; build it with "
                        f"`bash benchmarks/setup_baseline_env.sh`).")
    p.add_argument("--resume", action="store_true",
                   help="Skip if this run directory already has a result.")
    p.add_argument("--setup", action="store_true",
                   help="Clone/update the upstream repo, check dependencies, "
                        "and exit without running inference.")
    p.add_argument("--fetch-ckpt", action="store_true",
                   help="Download the default pretrained checkpoint if absent.")
    return p


def _open_run_dir(args: argparse.Namespace) -> Path:
    """Resolve --out-dir, creating an explicit path if it doesn't exist yet."""
    try:
        return resolve_run_dir(args.results_root, args.out_dir)
    except FileNotFoundError:
        looks_like_path = "/" in args.out_dir
        if not looks_like_path:
            raise SystemExit(
                f"Run directory '{args.out_dir}' not found under "
                f"{args.results_root}.\nStart one with `kgfm bench run`, or "
                f"pass --out-dir <path> to create a standalone directory."
            )
        path = Path(args.out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()


def run(spec: Baseline, argv: Optional[List[str]] = None) -> Optional[dict]:
    args = build_parser(spec).parse_args(argv)
    ckpt = args.ckpt or spec.default_ckpt

    if args.setup or args.fetch_ckpt:
        if args.setup:
            clone_or_update(spec, args.repo_dir)
            check_deps(spec, args.conda_env)
        if args.fetch_ckpt:
            fetch_ckpt(spec, ckpt)
        return None

    out_dir = _open_run_dir(args)
    logger = RunLogger(out_dir)
    out_path = out_dir / f"{spec.method.lower()}.json"

    if args.resume and out_path.is_file():
        logger.log(f"skip {spec.method} (resume: {out_path.name} exists)")
        return None
    logger.record_command(tag=spec.method, note=f"{spec.method} zero-shot inference")

    cpu_run = args.gpus.strip().lower() in ("null", "[]", "none")
    if cpu_run and not spec.cpu_supported:
        logger.log(f"!! {spec.method} cannot run on CPU: {spec.cpu_unsupported_reason}")
        logger.log(f"   Give it a GPU, in an env whose nvcc matches its torch:")
        logger.log(f"   {spec.command} --out-dir {out_dir.name} --conda-env <env> --gpus '[0]'")
        return None

    for label, path, fix in (
        ("repo", args.repo_dir, f"{spec.command} --setup"),
        ("prepared KG", args.kg_dir, "kgfm bench prep"),
    ):
        if not os.path.isdir(path):
            logger.log(f"!! {spec.method} {label} not found at {path}; run `{fix}`")
            return None
    if not os.path.isfile(ckpt):
        logger.log(
            f"!! {spec.method} checkpoint not found at {ckpt}; "
            f"pass --ckpt or run `{spec.command} --fetch-ckpt`"
        )
        return None

    data_root_abs = os.path.abspath(args.kg_dir)
    logger.log(f"[{spec.command}] patching {args.repo_dir}/{spec.package}/datasets.py")
    _inject_shim(spec, args.repo_dir, data_root_abs)
    _patch_pyg_version(spec, args.repo_dir)
    _inject_test_count(spec, args.repo_dir)

    python = envs.env_python(args.conda_env)
    config = _cpu_config(spec, args.repo_dir, args.config) if cpu_run else args.config
    if config != args.config:
        logger.log(f"[{spec.command}] CPU run: using {config} (Triton kernels off)")
    cmd = [
        python, "script/run.py",
        "-c", config,
        "--dataset", "ChEMBLCustom",
        "--gpus", args.gpus,
        "--epochs", "0",
        "--bpe", args.bpe,
        "--ckpt", os.path.abspath(ckpt),
    ]
    child_env = envs.cuda_build_env(args.conda_env)
    # ULTRA and MOTIF both JIT-build an extension *named* `rspmm` from
    # different sources. torch caches those by name, so sharing the default
    # cache means whichever ran last wins and a stale object file from the
    # other project gets linked in (seen in the wild as
    # "libcudart.so.12: cannot open shared object file").
    #
    # Keyed by repo *and* env: setting TORCH_EXTENSIONS_DIR suppresses the
    # `py3XX_cuYYY` subdirectory torch would otherwise add, so without the env
    # in the path, running the same baseline under two envs (different python
    # or CUDA) would collide in exactly the way we are trying to prevent.
    child_env["TORCH_EXTENSIONS_DIR"] = os.path.abspath(
        os.path.join(args.repo_dir, ".torch_extensions", args.conda_env)
    )
    if cpu_run:
        # `--gpus null` is not enough to get a CPU run: rspmm's loader compiles
        # its .cu sources whenever `torch.cuda.is_available()`, so on a box with
        # GPUs but no matching nvcc the build fails before inference starts.
        # Hiding the devices makes it take the CPU-only source list.
        child_env["CUDA_VISIBLE_DEVICES"] = ""

    logger.log(f"[{spec.command}] python={python} (conda env '{args.conda_env}')")
    logger.log(f"[{spec.command}] gpus={args.gpus}"
               + (" (CUDA hidden from the child)"
                  if child_env.get("CUDA_VISIBLE_DEVICES") == "" else ""))

    t0 = time.time()
    result = logger.run(
        cmd, step=spec.method.lower(), env=child_env,
        cwd=args.repo_dir, capture=True, tag=spec.method,
    )
    elapsed = time.time() - t0

    record = {
        "method": spec.method,
        "mode": "zero-shot",
        "ckpt": ckpt,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": parse_metrics(result.output) if result.ok else {},
        "data_root": data_root_abs,
        "config": args.config,
        "gpus": args.gpus,
        "python": python,
        "conda_env": args.conda_env,
    }
    if not result.ok:
        record["error"] = f"non-zero exit code {result.returncode}"
        if envs.missing_modules(DEPS, args.conda_env):
            record["hint"] = (
                f"dependencies missing in conda env '{args.conda_env}'; "
                f"run `ENV_NAME={args.conda_env} bash "
                f"benchmarks/setup_baseline_env.sh`"
            )
    out_path.write_text(json.dumps(record, indent=2))
    logger.log(f"[{spec.command}] wrote {out_path}")
    return record
