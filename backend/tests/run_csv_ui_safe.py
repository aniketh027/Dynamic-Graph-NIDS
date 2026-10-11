"""Fixed C3B schema tests or a synthetic-only C3A differential oracle.

Uses C3A's corrected accidental-access guard before application imports. No
arbitrary paths or pytest selections; native regular-file opens stay in arena.
"""
import argparse
import base64
from contextlib import redirect_stdout
import inspect
import json
from pathlib import Path
import sys
import tempfile
import os

from backend.tests.run_csv_compatibility_safe import (
    ROOT, install_guard, expect_denial, enumeration_self_checks,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differential", action="store_true")
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with tempfile.TemporaryDirectory(prefix="graphnids-c3b-safe-") as directory:
        arena = Path(directory).resolve()
        if arena.is_relative_to(ROOT):
            raise RuntimeError("Synthetic arena must be outside the repository")
        protected = set()
        violations, reject, check = install_guard(arena, protected_roots=protected)
        original_temp = tempfile.tempdir
        tempfile.tempdir = str(arena)
        with redirect_stdout(sys.stderr):
            for path, mode in ((ROOT / "data/raw/csv/prohibited.csv", "rb"),
                               (ROOT / "PROJECT_HANDOFF.md", "wb"),
                               (ROOT / "backend/prohibited.tmp", "wb")):
                expect_denial(violations, lambda: path.open(mode))
            enumeration_self_checks(arena, protected, violations)
        violations.clear()

        class NoScientificImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("src.training", "src.pruning", "backend.app.ml")):
                    reject("scientific/model import")
                return None
        sys.meta_path.insert(0, NoScientificImports())
        from backend.app.services import csv_replay, csv_compatibility as compat, inference, artifacts, replay
        from src.preprocessing import transforms

        def forbidden(*args, **kwargs):
            reject("transform/model/training/pruning/evaluation operation")
        for name in ("load_transform", "transform_batch", "apply_transforms"):
            setattr(csv_replay, name, forbidden)
        for name, value in tuple(vars(transforms).items()):
            if inspect.isfunction(value) and value.__module__ == transforms.__name__:
                setattr(transforms, name, forbidden)
        transforms.FitTransformState.from_dict = forbidden
        artifacts.discover = inference.discover = inference.load_frozen = forbidden
        replay.ReplayService._get_adapter = forbidden
        native_open = compat._windows_handle

        def confined_open(path, *, directory=False, create=False):
            if not directory:
                check(path, True)
            return native_open(path, directory=directory, create=create)
        compat._windows_handle = confined_open
        if violations:
            raise RuntimeError("Unexpected guard violations during imports")
        expect_denial(violations, lambda: csv_replay.load_transform(ROOT))
        expect_denial(violations, lambda: inference.load_frozen(ROOT, None))
        expect_denial(violations, lambda: transforms.FitTransformState.from_dict({}))
        violations.clear()
        try:
            if args.differential:
                request = sys.stdin.read(32 * 1024 * 1024 + 1)
                if len(request) > 32 * 1024 * 1024:
                    raise ValueError("Oracle request exceeds bound")
                cases = json.loads(request)
                if not isinstance(cases, list) or len(cases) > 200:
                    raise ValueError("Oracle case list exceeds bound")
                results = []
                for index, case in enumerate(cases):
                    if not isinstance(case, dict) or set(case) != {"source_base64"}:
                        raise ValueError("Only synthetic byte cases are supported")
                    payload = base64.b64decode(case["source_base64"], validate=True)
                    if len(payload) > csv_replay.MAX_BYTES + 1:
                        raise ValueError("Oracle case exceeds source bound")
                    source, output = arena / f"source-{index}.csv", arena / f"output-{index}.csv"
                    source.write_bytes(payload)
                    try:
                        manifest = compat.prepare(source, output)
                    except compat.CompatibilityError:
                        results.append({"accepted": False})
                    else:
                        results.append({"accepted": True,
                                        "prepared_base64": base64.b64encode(output.read_bytes()).decode("ascii"),
                                        "record_count": manifest["output_record_count"],
                                        "source_sha256": manifest["source"]["sha256"],
                                        "prepared_sha256": manifest["output"]["sha256"]})
                if violations:
                    raise RuntimeError("Unexpected differential guard violations")
                print(json.dumps(results, separators=(",", ":")))
                code = 0
            else:
                import pytest
                code = pytest.main(["-q", "-p", "no:cacheprovider", "--confcutdir=backend/tests",
                                    "--log-file", str(arena / "pytest.log"), "--basetemp", str(arena / "pytest"),
                                    "backend/tests/test_csv_ui_schema.py"])
            print("C3B unexpected guard violations:", len(violations), file=sys.stderr)
            return int(code) if not violations else 1
        finally:
            tempfile.tempdir = original_temp


if __name__ == "__main__":
    raise SystemExit(main())
