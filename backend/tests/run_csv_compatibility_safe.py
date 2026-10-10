"""Run only C3A synthetic tests and an optional fixed C2A rejection subset.

Python accidental-access guard, not a native-code sandbox. No arbitrary pytest
arguments are accepted. Native CSV opens are separately confined to the test
arena. Use the repository .venv with -B; no model or scientific data is needed.
"""
import argparse
import inspect
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PARSER_REGRESSIONS = tuple(f"backend/tests/test_replay.py::{name}" for name in (
    "test_rejected_headers_before_inference",
    "test_empty_malformed_and_archive_content",
    "test_strict_numeric_rejects_text_and_overflow",
))


def install_guard(arena, *, protected_roots=()):
    violations = []

    def reject(reason):
        violations.append(reason)
        raise PermissionError("C3A safe runner rejected " + reason)

    def check(value, writing, *, follow_leaf=True):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return
        lexical = Path(os.fsdecode(value)).absolute()
        resolved = lexical.resolve() if follow_leaf else lexical.parent.resolve() / lexical.name
        if writing:
            if not all(path.is_relative_to(arena) for path in (lexical, resolved)):
                reject("write outside the external synthetic arena")
        else:
            for path in (lexical, resolved):
                if any(path.is_relative_to(root) for root in protected_roots):
                    reject("protected synthetic root read")
                if path.is_relative_to(ROOT):
                    relative = path.relative_to(ROOT)
                    first = relative.parts[0].lower() if relative.parts else ""
                    if (first in {"data", "results", "logs", "release", "checkpoints", ".git"}
                            or first.startswith(("e10-", "run_e10_"))
                            or path.suffix.lower() in {".csv", ".npy", ".npz", ".pt", ".pth", ".png", ".jpg", ".zip"}):
                        reject("protected repository read")

    def hook(event, args):
        if event in {"os.scandir", "os.listdir"}:
            # Both APIs emit (None,) for omitted/None paths on Windows. That
            # means cwd, not an exemption. Windows rejects native fd arguments
            # before auditing; reject fd/unknown forms at this boundary too.
            if len(args) != 1:
                reject("unsupported directory enumeration arguments")
            value = args[0]
            if value is not None and not isinstance(value, (str, bytes, os.PathLike)):
                reject("unsupported directory enumeration path")
            before = len(violations)
            try:
                value = os.getcwd() if value is None else os.fsdecode(value)
                if "\x00" in value:
                    raise ValueError("embedded null")
                # pathlib retains device/extended namespace prefixes, so those
                # paths cannot safely share the ordinary protected-root check.
                if os.name == "nt" and value.replace("/", "\\").startswith(("\\\\?\\", "\\\\.\\", "\\??\\")):
                    reject("unsupported directory enumeration namespace")
                check(value, False)
            except (OSError, TypeError, ValueError, RuntimeError):
                if len(violations) == before:
                    reject("unresolvable directory enumeration path")
                raise
        elif event == "open":
            mode, flags = args[1:3]
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                       or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            check(args[0], writing)
        elif event in {"os.remove", "os.rmdir"}:
            # Removing a junction removes that directory entry, not its target.
            check(args[0], True, follow_leaf=False)
        elif event in {"os.mkdir", "os.chmod", "os.utime"}:
            check(args[0], True)
        elif event in {"os.rename", "os.link"}:
            check(args[0], True)
            check(args[1], True)
        elif event == "os.symlink":
            check(args[1], True)
        elif event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
            reject("external process execution")
    sys.addaudithook(hook)
    return violations, reject, check


def expect_denial(violations, operation):
    before = len(violations)
    try:
        operation()
    except PermissionError:
        if len(violations) != before + 1:
            raise RuntimeError("Guard denial was not recorded")
    else:
        raise RuntimeError("Guard silently permitted a prohibited operation")


def enumeration_self_checks(arena, protected_roots, violations):
    protected = arena / "protected-enumeration"
    allowed = arena / "allowed-enumeration"
    protected.mkdir()
    allowed.mkdir()
    (protected / "private-marker").write_text("synthetic")
    (allowed / "safe-marker").write_text("synthetic")
    alias = arena / "enumeration-junction"
    import _winapi
    try:
        _winapi.CreateJunction(str(protected), str(alias))
    except OSError as exc:
        if exc.winerror != 1314:
            raise
        print("Enumeration junction unavailable: Windows privilege error 1314")
        alias = None
    protected_roots.add(protected)
    old_cwd = Path.cwd()
    before = len(violations)
    allowed_count = 0

    def enumerate_names(api, *arguments):
        result = api(*arguments)
        if api is os.scandir:
            with result:
                return sorted(os.fsdecode(entry.name) for entry in result)
        return sorted(os.fsdecode(name) for name in result)

    class BadPath:
        def __fspath__(self):
            return 1

    try:
        for api in (os.listdir, os.scandir):
            os.chdir(arena)
            for path in (str(protected), protected, os.fsencode(protected),
                         protected.name, str(protected).upper(),
                         protected / ".." / protected.name,
                         "\\\\?\\" + str(protected), "\\\\.\\" + str(protected)):
                expect_denial(violations, lambda: enumerate_names(api, path))
            if alias is not None:
                for path in (str(alias), alias, os.fsencode(alias), alias.name):
                    expect_denial(violations, lambda: enumerate_names(api, path))
            os.chdir(protected)
            for operation in (lambda: enumerate_names(api),
                              lambda: enumerate_names(api, None),
                              lambda: enumerate_names(api, ".")):
                expect_denial(violations, operation)
            # Emitted boundary probes cannot reach any OS directory. Native
            # Windows fd arguments fail before an event; do not count that as
            # a guard denial. Malformed audit forms must register a denial.
            for arguments in ((0,), (object(),), (BadPath(),), ("bad\x00path",), (), (str(protected), None)):
                expect_denial(violations, lambda: sys.audit("os." + api.__name__, *arguments))
            os.chdir(allowed)
            for arguments in ((str(allowed),), (allowed,), (os.fsencode(allowed),), (".",), (), (None,)):
                if enumerate_names(api, *arguments) != ["safe-marker"]:
                    raise RuntimeError("Permitted synthetic traversal failed")
                allowed_count += 1
            # Discovery must be allowed to enumerate the repository root itself.
            sys.audit("os." + api.__name__, str(ROOT))
        if len(violations) != before + 2 * (17 + (4 if alias is not None else 0)):
            raise RuntimeError("Enumeration self-check ledger mismatch")
        print(f"Enumeration self-checks: {len(violations) - before} recorded denials; "
              f"{allowed_count} permitted traversals; junction tested: {alias is not None}")
    finally:
        os.chdir(old_cwd)
        protected_roots.remove(protected)  # Only the disposable fixture expires.
        if alias is not None:
            alias.rmdir()  # Remove the entry, without traversing its target.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-regressions", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="graphnids-c3a-safe-") as directory:
        arena = Path(directory).resolve()
        if arena.is_relative_to(ROOT):
            raise RuntimeError("Temporary storage must be outside the repository")
        protected_roots = set()  # Additional disposable roots for self-checks only.
        violations, reject, check = install_guard(arena, protected_roots=protected_roots)
        original_temp = tempfile.tempdir
        tempfile.tempdir = str(arena)  # pytest capture files also stay in arena

        # Prove denials before any OS access. These are planned self-checks.
        for path, mode in ((ROOT / "data/raw/csv/prohibited.csv", "rb"),
                           (ROOT / "PROJECT_HANDOFF.md", "wb"),
                           (ROOT / "backend/prohibited.tmp", "wb")):
            expect_denial(violations, lambda: path.open(mode))
        assert len(violations) == 3
        enumeration_self_checks(arena, protected_roots, violations)
        violations.clear()
        print("Guard self-checks: 3 planned denials before OS access")

        import pytest
        from backend.app.services import csv_compatibility as compat, csv_replay
        from src.preprocessing import transforms

        def forbidden(*args, **kwargs):
            reject("transform/model/training/pruning/evaluation operation")

        if not args.parser_regressions:
            class NoScientificImports:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith(("src.training", "src.pruning", "backend.app.ml",
                                            "backend.app.main", "backend.app.services.inference",
                                            "backend.app.services.artifacts", "backend.app.services.replay")):
                        reject("scientific/model/server import")
                    return None
            sys.meta_path.insert(0, NoScientificImports())

        for name in ("load_transform", "transform_batch", "apply_transforms"):
            setattr(csv_replay, name, forbidden)
        for name, value in tuple(vars(transforms).items()):
            if inspect.isfunction(value) and value.__module__ == transforms.__name__:
                setattr(transforms, name, forbidden)
        transforms.FitTransformState.from_dict = forbidden
        native_open = compat._windows_handle

        def confined_open(path, *, directory=False, create=False):
            # Parent attribute handles may be outside arena; regular-file native
            # opens must stay in arena. _pin checks and rejects reparse ancestors.
            if not directory:
                check(path, True)
            return native_open(path, directory=directory, create=create)
        compat._windows_handle = confined_open

        requested = ["backend/tests/test_csv_compatibility.py"]
        if args.parser_regressions:
            # Inspected module imports define functions/classes only; audit guard
            # is active before import. The selected requests reject during parse,
            # before any state read, transformation or lazy model initialization.
            from backend.tests import test_replay
            import torch
            from backend.app.ml.application_inference import ApplicationInferenceAdapter
            from backend.app.services import artifacts, inference, replay
            for module in tuple(sys.modules.values()):
                name = getattr(module, "__name__", "")
                if name.startswith(("src.training.", "src.pruning.")):
                    for attribute, value in tuple(vars(module).items()):
                        if inspect.isfunction(value) and value.__module__.startswith(("src.training.", "src.pruning.")):
                            setattr(module, attribute, forbidden)
            artifacts.discover = inference.discover = inference.load_frozen = forbidden
            replay.ReplayService._get_adapter = forbidden
            ApplicationInferenceAdapter.__init__ = forbidden
            ApplicationInferenceAdapter.infer = forbidden
            ApplicationInferenceAdapter.logits = forbidden
            torch.load = forbidden
            torch.optim.Optimizer.__init__ = forbidden
            torch.Tensor.backward = torch.autograd.backward = torch.autograd.grad = forbidden
            torch.nn.Module.__call__ = forbidden
            # test_replay imported this transform alias; keep it prohibited too.
            test_replay.apply_transforms = forbidden
            requested = list(PARSER_REGRESSIONS)
        if violations:
            raise RuntimeError("Unexpected guard violations during imports")
        expect_denial(violations, lambda: csv_replay.load_transform(ROOT))
        expect_denial(violations, lambda: transforms.FitTransformState.from_dict({}))
        if args.parser_regressions:
            expect_denial(violations, lambda: torch.load(arena / "prohibited.pt"))
        print("Scientific guard self-checks:", len(violations), "recorded denials")
        violations.clear()
        try:
            code = pytest.main(["-q", "-p", "no:cacheprovider", "--confcutdir=backend/tests",
                                "--log-file", str(arena / "pytest.log"),
                                "--basetemp", str(arena / "pytest"), *requested])
        finally:
            tempfile.tempdir = original_temp
        print("C3A unexpected guard violations:", len(violations))
        return int(code) if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
