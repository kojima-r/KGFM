"""Run directories and step logging.

A benchmark *run* is a directory under ``benchmarks/results/chembl/`` named by
UTC timestamp (plus an optional label). Everything about the run lands in it:
meta.json, one JSON per method, table.md, checkpoints, and logs.

``RunLogger`` reproduces what the old shell `run_step` did — stream a child's
output to the terminal while also appending it to both ``run.log`` and a
per-step ``<step>.log`` — and returns the exit status rather than raising, so
one failing method never takes the rest of the run down with it.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, TextIO


class _Tee:
    """Minimal write-only stream fan-out (terminal + step log + run log)."""

    def __init__(self, streams: Sequence[TextIO]):
        self._streams = list(streams)

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            with contextlib.suppress(Exception):
                stream.flush()

    def isatty(self) -> bool:
        return False


def current_command() -> str:
    """This process's command line, as something you could paste back.

    ``sys.argv[0]`` is an absolute path — the console script, or
    ``.../kgfm/__main__.py`` under ``python -m``. Neither is what the user
    typed, so both are normalised back to the command name.
    """
    argv = list(sys.argv)
    if not argv:
        return ""
    exe = Path(argv[0]).name
    if exe == "__main__.py":
        head = "python -m kgfm"
    elif exe.endswith(".py"):
        head = f"python {argv[0]}"
    else:
        head = exe
    return shlex.join([head, *argv[1:]]) if hasattr(shlex, "join") else " ".join(
        [head, *(shlex.quote(a) for a in argv[1:])]
    )


def parent_command() -> Optional[str]:
    """The command that launched this one, when it is a shell wrapper.

    Best-effort and Linux-only: lets the report say the run came from
    ``bash benchmarks/run_chembl.sh`` rather than only showing the `kgfm`
    invocation that script produced.
    """
    try:
        with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
            parts = [p.decode(errors="replace") for p in f.read().split(b"\0") if p]
    except OSError:
        return None
    if not parts:
        return None
    # Only surface it when it is recognisably one of our wrappers; a bare
    # interactive shell as the parent is noise.
    joined = " ".join(parts)
    if any(name in joined for name in ("run_chembl", "resume_chembl", "setup_")):
        return joined
    return None


@dataclass
class StepResult:
    returncode: int
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class RunLogger:
    """Timestamped logging into a run directory."""

    def __init__(self, out_dir: os.PathLike | str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "run.log"

    # ---- command provenance -------------------------------------------
    #
    # Every command that writes into a run directory appends a line here, so
    # the report can show exactly how the run was produced. A JSONL file
    # rather than a field in meta.json because several independent commands
    # (bench run, kgfm-ultra, kgfm-motif, viz, report) contribute over time.
    COMMANDS_FILE = "commands.jsonl"

    def record_command(
        self,
        *,
        kind: str = "invocation",
        tag: Optional[str] = None,
        command: Optional[str] = None,
        note: Optional[str] = None,
    ) -> None:
        """Append one command to the run's provenance log."""
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "command": command if command is not None else current_command(),
            "cwd": os.getcwd(),
        }
        if tag:
            entry["tag"] = tag
        if note:
            entry["note"] = note
        if kind == "invocation":
            parent = parent_command()
            if parent:
                entry["parent"] = parent
        with open(self.out_dir / self.COMMANDS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log(self, message: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Capture an *in-process* step's stdout/stderr into the logs.

        The subprocess path (`run`) tees by reading the child's pipe; steps
        that stay in this interpreter need their prints redirected instead,
        or run.log would only ever record the steps that forked.
        """
        self.log(f"==> {name}")
        old_out, old_err = sys.stdout, sys.stderr
        with open(self.out_dir / f"{name}.log", "w", encoding="utf-8") as sf, \
                open(self.log_path, "a", encoding="utf-8") as rf:
            tee = _Tee([old_out, sf, rf])
            sys.stdout = sys.stderr = tee            # type: ignore[assignment]
            try:
                yield
            finally:
                tee.flush()
                sys.stdout, sys.stderr = old_out, old_err

    def run(
        self,
        cmd: Sequence[str],
        *,
        step: str,
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
        capture: bool = False,
        tag: Optional[str] = None,
    ) -> StepResult:
        """Run ``cmd``, teeing its output. Never raises on a non-zero exit.

        ``capture=True`` additionally returns the output, which ULTRA / MOTIF
        need because their metrics are only available as stdout text.
        """
        self.log(f"==> {step}")
        argv = [str(c) for c in cmd]
        self.record_command(
            kind="step", tag=tag or step,
            command=" ".join(shlex.quote(a) for a in argv),
        )
        chunks: List[str] = []
        step_log = self.out_dir / f"{step}.log"
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, cwd=cwd,
            )
        except OSError as exc:
            self.log(f"    !! {step} could not start: {exc}")
            return StepResult(127)

        with open(step_log, "w", encoding="utf-8") as sf, \
                open(self.log_path, "a", encoding="utf-8") as rf:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                sf.write(line)
                rf.write(line)
                if capture:
                    chunks.append(line)
        rc = proc.wait()
        if rc != 0:
            self.log(f"    !! {step} exited {rc} — continuing")
        return StepResult(rc, "".join(chunks))


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def sanitize_label(label: str) -> str:
    """Keep a run label safe as a directory-name postfix."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in label)


def create_run_dir(results_root: os.PathLike | str, label: str = "") -> Path:
    """Make a fresh ``<ts>[_<label>]`` run dir and repoint ``latest`` at it."""
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    label = sanitize_label(label or "")
    name = utc_stamp() + (f"_{label}" if label else "")
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(name)
    return out_dir


def resolve_run_dir(results_root: os.PathLike | str, target: str) -> Path:
    """Resolve a resume target into an existing run dir.

    Accepts ``latest``, a bare timestamp (``20260507T155834Z``), or a path.
    Symlinks are followed so the returned name is the real directory.
    """
    root = Path(results_root)
    candidate = Path(target) if ("/" in target or target.startswith("/")) else root / target
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resume target not found: {candidate}")
    return candidate.resolve()


def read_commands(out_dir: os.PathLike | str) -> List[dict]:
    """Every recorded command for a run, oldest first."""
    path = Path(out_dir) / RunLogger.COMMANDS_FILE
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def run_timestamp(out_dir: Path) -> str:
    """Recover the bare UTC timestamp from a run dir name.

    The timestamp itself contains no underscore, so anything from the first
    underscore on is the label the wrapper attached.
    """
    return out_dir.name.split("_", 1)[0]
