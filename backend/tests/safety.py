"""Accidental-access guard for C1 checks; no official datasets or experiments.

This is a Python audit boundary, not a sandbox against arbitrary native code.
Use python -B so imports do not write scientific caches.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_READS = ("data", "results/stage_e11", "results/stage_e12b", "logs/stage_e11", "logs/stage_e12b")
FROZEN_AREAS = ("src", "scripts", "tests", "results", "logs", "release", "checkpoints", "data", ".git")
violations: list[str] = []


def install_guard(*, allowed_read_files: tuple[Path, ...] = ()) -> list[str]:
    allowed = {p.resolve() for p in allowed_read_files}
    def forbidden(event: str, target) -> None:
        if isinstance(target, (int, type(None))):
            return
        path = Path(os.fsdecode(target)).resolve()
        if not path.is_relative_to(ROOT):
            return
        relative = path.relative_to(ROOT).as_posix()
        top = relative.split("/")[0]
        protected = top.startswith("e10-") or top.startswith("run_e10_")
        if event == "read":
            reject = path not in allowed and any(relative == prefix or relative.startswith(prefix + "/") for prefix in FORBIDDEN_READS)
        else:
            reject = top in FROZEN_AREAS or protected or top in {"AGENTS.md", "PROJECT_HANDOFF.md"} or top.startswith("requirements")
        if reject:
            violations.append(f"{event}: {relative}")
            raise PermissionError("C1 guard rejected frozen-state write or official-data/test-evidence access")

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            mode = args[1]
            writing = (isinstance(mode, str) and any(c in mode for c in "wa+x")) or (
                isinstance(args[2], int) and bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            forbidden("write" if writing else "read", args[0])
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.rename"}:
            forbidden("write", args[0])
            if event == "os.rename":
                forbidden("write", args[1])

    sys.addaudithook(hook)
    return violations
