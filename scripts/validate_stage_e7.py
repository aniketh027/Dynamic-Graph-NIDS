"""Run bounded Stage E.7 smoke checks and write compact contract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import resource_pilot as e6  # noqa: E402
from src.training import harness as e7  # noqa: E402
from src.training import validation as evidence  # noqa: E402


def _run_pytest(project_root: Path, args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *args],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    passed = re.findall(r"(\d+) passed", completed.stdout)
    skipped = re.findall(r"(\d+) skipped", completed.stdout)
    return {
        "command": "python -m pytest -q " + " ".join(args),
        "exit_code": completed.returncode,
        "passed": int(passed[-1]) if passed else 0,
        "skipped": int(skipped[-1]) if skipped else 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _smoke_task(project_root: Path, task: str, checkpoint: Path) -> dict[str, Any]:
    config = e7.TrainingConfig(
        task=task,
        learning_rate=3e-4,
        seed=2025,
        batch_size=8,
        max_epochs=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    result = e7.run_training(
        config,
        project_root=project_root,
        checkpoint_path=checkpoint,
        device="auto",
    )
    metadata, validation, model = e7.evaluate_saved_validation_checkpoint(
        config,
        project_root=project_root,
        checkpoint_path=checkpoint,
        device="auto",
    )
    device = next(model.parameters()).device
    dataset = d4.LazyGraphDataset(
        task,
        "validation",
        processed_root=project_root / "data" / "processed" / "preprocessed",
    )
    batch = next(
        e7.iter_production_batches(
            dataset,
            task=task,
            split="validation",
            batch_size=8,
            epoch_seed=None,
            max_batches=1,
        )
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    expected_shape = [8] if task == "E3" else [8, 15]
    reloaded_matches = bool(
        np.isclose(validation.loss, metadata["validation_loss"], rtol=1e-5, atol=1e-6)
        and np.isclose(
            validation.selection_value,
            metadata["selection_value"],
            rtol=1e-5,
            atol=1e-6,
        )
    )
    checks = {
        "task": task,
        "batch_size": 8,
        "train_batches": result.history[0]["train_batches"],
        "validation_batches": result.history[0]["validation_batches"],
        "train_examples": result.history[0]["train_examples"],
        "validation_examples": result.history[0]["validation_examples"],
        "output_shape": list(logits.shape),
        "expected_output_shape": expected_shape,
        "loss_finite": bool(
            np.isfinite(result.history[0]["train_loss"])
            and np.isfinite(result.history[0]["validation_loss"])
        ),
        "gradient_and_optimizer_step_completed": result.parameter_update_observed,
        "best_checkpoint_selected": result.best_epoch == 1,
        "checkpoint_reloaded": True,
        "reloaded_validation_state_matches": reloaded_matches,
    }
    checks["status"] = (
        "PASS"
        if all(
            (
                checks["train_batches"] == 1,
                checks["validation_batches"] == 1,
                checks["train_examples"] == 8,
                checks["validation_examples"] == 8,
                checks["output_shape"] == expected_shape,
                checks["loss_finite"],
                checks["gradient_and_optimizer_step_completed"],
                checks["best_checkpoint_selected"],
                checks["reloaded_validation_state_matches"],
            )
        )
        else "FAIL"
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage E.7 training harness")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_e7/htgnn_training_harness_validation.json"),
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if not args.run_tests:
        parser.error("--run-tests is required when writing canonical E.7 evidence")
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output

    frozen_hashes = e6.require_frozen_sources(project_root)
    with tempfile.TemporaryDirectory(prefix="stage_e7_smoke_") as temporary:
        temp_root = Path(temporary)
        smoke = [
            _smoke_task(project_root, task, temp_root / f"{task}.pt")
            for task in e7.TASKS
        ]
    temporary_removed = not temp_root.exists()
    for record in smoke:
        record["checkpoint_removed_after_smoke"] = temporary_removed
        if not temporary_removed:
            record["status"] = "FAIL"
    integrity_result = d4.verify_frozen_integrity(project_root)
    integrity = {
        "stage_c": "PASS" if integrity_result["stage_c_sha256_matches"] else "FAIL",
        "stage_c_matched": integrity_result["stage_c_artifact_count"],
        "stage_c_expected": integrity_result["stage_c_expected_count"],
        "raw_csv": "PASS" if integrity_result["raw_csv_sha256_matches"] else "FAIL",
        "raw_csv_matched": integrity_result["raw_csv_count"],
        "raw_csv_expected": integrity_result["raw_csv_expected_count"],
    }
    verification: dict[str, Any] = {
        "focused_e7": _run_pytest(
            project_root, ["tests/test_training_harness.py"]
        ),
        "frozen_e2_e6_regression": _run_pytest(
            project_root,
            [
                "tests/test_pyg_adapter.py",
                "tests/test_e2_provenance_cache.py",
                "tests/test_feature_encoder.py",
                "tests/test_hierarchical_encoder.py",
                "tests/test_htgnn_classifier.py",
                "tests/test_resource_pilot.py",
            ],
        ),
        "complete_project_suite": _run_pytest(project_root, []),
    }
    verification["status"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in verification.values())
        else "FAIL"
    )
    report = evidence.build_report(
        frozen_source_sha256=frozen_hashes,
        smoke_validation=smoke,
        integrity=integrity,
        verification=verification,
    )
    payload = evidence.report_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    physical_hash = hashlib.sha256(payload).hexdigest()
    output.with_suffix(".sha256").write_text(
        f"{physical_hash}  {output.name}\n", encoding="ascii"
    )
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"physical_sha256={physical_hash}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
