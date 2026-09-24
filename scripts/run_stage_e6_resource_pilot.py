"""Run the bounded real-data Stage E.6 HTGNN resource pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import resource_pilot as e6  # noqa: E402
from src.graph import schema as d2  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_pytest(project_root: Path, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    passed_matches = re.findall(r"(\d+) passed", completed.stdout)
    skipped_matches = re.findall(r"(\d+) skipped", completed.stdout)
    return {
        "command": "python -m pytest -q " + " ".join(args),
        "exit_code": completed.returncode,
        "passed": int(passed_matches[-1]) if passed_matches else 0,
        "skipped": int(skipped_matches[-1]) if skipped_matches else 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _write_contract_report(report: dict[str, Any], output: Path) -> None:
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    report["report_sha256"] = d2.canonical_sha256(unhashed)
    payload = d2.canonical_json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    output.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {output.name}\n",
        encoding="ascii",
    )


def _write_observations(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded Stage E.6 real-data CUDA forward measurements"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--batch-size", type=int, default=e6.DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--warmup-iterations", type=int, default=e6.DEFAULT_WARMUP_ITERATIONS
    )
    parser.add_argument(
        "--measured-iterations", type=int, default=e6.DEFAULT_MEASURED_ITERATIONS
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    contract_path = (
        project_root / "results" / "stage_e6" / "htgnn_resource_pilot_contract.json"
    )
    observations_path = (
        project_root
        / "results"
        / "stage_e6"
        / "htgnn_resource_pilot_observations.json"
    )
    config = e6.PilotConfig(
        batch_size=args.batch_size,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
    )
    contract, observations = e6.run_real_pilot(project_root, config)
    integrity = d4.verify_frozen_integrity(project_root)
    contract["stage_c_integrity"] = {
        "status": "PASS" if integrity["stage_c_sha256_matches"] else "FAIL",
        "matched": integrity["stage_c_artifact_count"],
        "expected": integrity["stage_c_expected_count"],
    }
    contract["raw_data_integrity"] = {
        "status": "PASS" if integrity["raw_csv_sha256_matches"] else "FAIL",
        "matched": integrity["raw_csv_count"],
        "expected": integrity["raw_csv_expected_count"],
    }
    integrity_passed = (
        contract["stage_c_integrity"]["status"] == "PASS"
        and contract["raw_data_integrity"]["status"] == "PASS"
    )
    contract["overall_status"] = (
        "PASS"
        if contract["overall_status"] == "PASS" and integrity_passed
        else "FAIL"
    )
    _write_observations(observations, observations_path)
    _write_contract_report(contract, contract_path)

    if args.run_tests:
        verification = {
            "focused_e6": _run_pytest(
                project_root, ["tests/test_resource_pilot.py"]
            ),
            "frozen_e2_e5_regression": _run_pytest(
                project_root,
                [
                    "tests/test_pyg_adapter.py",
                    "tests/test_feature_encoder.py",
                    "tests/test_hierarchical_encoder.py",
                    "tests/test_htgnn_classifier.py",
                ],
            ),
            "complete_project_suite": _run_pytest(project_root, []),
        }
        verification["status"] = (
            "PASS"
            if all(item["status"] == "PASS" for item in verification.values())
            else "FAIL"
        )
        contract["verification"] = verification
        contract["overall_status"] = (
            "PASS"
            if contract["overall_status"] == "PASS"
            and verification["status"] == "PASS"
            else "FAIL"
        )
        _write_contract_report(contract, contract_path)

    print(f"overall_status={contract['overall_status']}")
    print(f"report_sha256={contract['report_sha256']}")
    print(f"physical_sha256={_sha256(contract_path)}")
    print(f"contract_output={contract_path}")
    print(f"observations_output={observations_path}")
    return 0 if contract["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
