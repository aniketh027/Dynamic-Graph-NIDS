"""Validate Stage E.9 and write compact deterministic contract evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import schema as d2  # noqa: E402
from src.training import baseline_integration as integration  # noqa: E402
from src.training import baselines  # noqa: E402
from src.training import batch_size_pilot as e9  # noqa: E402
from src.training import harness as e7  # noqa: E402


E9_SOURCE_PATHS = (
    "PROJECT_HANDOFF.md",
    "src/training/baselines.py",
    "src/training/baseline_integration.py",
    "src/training/batch_size_pilot.py",
    "src/training/harness.py",
    "src/training/optimization_pilot.py",
    "scripts/run_stage_e9_batch_size_pilot.py",
    "scripts/validate_stage_e9.py",
    "tests/test_stage_e9_baselines.py",
    "tests/test_stage_e9_batch_size_pilot.py",
    "tests/test_stage_e9_training_integration.py",
    "docs/stage_e9_baselines_batch_size_pilot.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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
    passed = re.findall(r"(\d+) passed", completed.stdout)
    skipped = re.findall(r"(\d+) skipped", completed.stdout)
    failed = re.findall(r"(\d+) failed", completed.stdout)
    return {
        "command": "python -m pytest -q " + " ".join(args),
        "exit_code": completed.returncode,
        "passed": int(passed[-1]) if passed else 0,
        "failed": int(failed[-1]) if failed else 0,
        "skipped": int(skipped[-1]) if skipped else 0,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _baseline_smoke(
    project_root: Path, *, kind: str, task: str, device_name: str
) -> dict[str, Any]:
    e7.seed_everything(2025)
    processed = project_root / "data" / "processed" / "preprocessed"
    validation_dataset = d4.LazyGraphDataset(
        task, "validation", processed_root=processed
    )
    config = e7.TrainingConfig(
        task=task,
        learning_rate=3e-4,
        seed=2025,
        batch_size=4,
        max_epochs=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    expected_parameters = baselines.EXPECTED_PARAMETER_COUNTS[kind][task]
    with tempfile.TemporaryDirectory(prefix=f"stage-e9-{kind}-{task}-") as temporary:
        temporary_root = Path(temporary)
        checkpoint = temporary_root / "best.pt"
        log = temporary_root / "run.jsonl"
        result = e7.run_training(
            config,
            project_root=project_root,
            checkpoint_path=checkpoint,
            log_path=log,
            device=device_name,
            model_family=kind,
        )
        records = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
        ]
        metadata, validation, first = e7.evaluate_saved_validation_checkpoint(
            config,
            project_root=project_root,
            checkpoint_path=checkpoint,
            device=device_name,
            model_family=kind,
        )
        second = e7.create_model(task, kind).to(device_name)
        e7.load_checkpoint(
            checkpoint,
            model=second,
            expected_task=task,
            expected_config=config,
            map_location=device_name,
        )
        validation_batch = next(
            e7.iter_production_batches(
                validation_dataset,
                task=task,
                split="validation",
                batch_size=4,
                epoch_seed=None,
                max_batches=1,
            )
        ).to(device_name)
        first.eval()
        second.eval()
        with torch.no_grad():
            first_output = first(validation_batch)
            second_output = second(validation_batch)
        output_reproduced = torch.allclose(
            first_output, second_output, rtol=1e-6, atol=1e-7
        )
        checkpoint_written = checkpoint.is_file() and checkpoint.stat().st_size > 0
        jsonl_written = log.is_file() and len(records) == 1
    checks = {
        "baseline": kind,
        "task": task,
        "device": device_name,
        "model_family": metadata["model_family"],
        "model_contract_version": metadata["model_contract_version"],
        "model_input_contract": metadata["model_input_contract"],
        "parameter_count": metadata["parameter_count"],
        "expected_parameter_count": expected_parameters,
        "train_batches": records[0]["train_batches"],
        "train_examples": records[0]["train_examples"],
        "validation_batches": validation.batches,
        "validation_examples": validation.examples,
        "train_loss_finite": math.isfinite(records[0]["train_loss"]),
        "validation_loss_finite": math.isfinite(validation.loss),
        "parameter_update_observed": result.parameter_update_observed,
        "validation_selection_name": validation.selection_name,
        "validation_selection_value_finite": math.isfinite(
            float(validation.selection_value)
        ),
        "early_stopping_reused": result.best_epoch == 1,
        "strict_state_dict_reload": True,
        "reloaded_output_reproduced": bool(output_reproduced),
        "checkpoint_written": checkpoint_written,
        "jsonl_log_written": jsonl_written,
        "jsonl_epoch_records": len(records),
        "test_accessed": False,
    }
    checks["status"] = (
        "PASS"
        if all(
            (
                checks["parameter_count"] == checks["expected_parameter_count"],
                checks["train_batches"] == 1,
                checks["train_examples"] == 4,
                checks["validation_batches"] == 1,
                checks["validation_examples"] == 4,
                checks["train_loss_finite"],
                checks["validation_loss_finite"],
                checks["parameter_update_observed"],
                checks["validation_selection_name"] == e7.SELECTION_NAMES[task],
                checks["validation_selection_value_finite"],
                checks["early_stopping_reused"],
                checks["model_family"] == kind,
                checks["checkpoint_written"],
                checks["jsonl_log_written"],
                checks["jsonl_epoch_records"] == 1,
                checks["reloaded_output_reproduced"],
            )
        )
        else "FAIL"
    )
    return checks


def _write_contract(report: dict[str, Any], output: Path) -> None:
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    report["report_sha256"] = d2.canonical_sha256(unhashed)
    payload = d2.canonical_json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    with output.with_suffix(".sha256").open(
        "w", encoding="ascii", newline="\n"
    ) as handle:
        handle.write(f"{hashlib.sha256(payload).hexdigest()}  {output.name}\n")


def _load_validated_observations(
    observations_path: Path,
    *,
    project_root: Path,
    verify_live_device: bool,
) -> Mapping[str, Any]:
    """Fail closed on full observation semantics before evidence work begins."""

    try:
        observations = json.loads(observations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise e9.BatchSizePilotError(
            f"cannot read Stage E.9 observations for evidence: {exc}"
        ) from exc
    if not isinstance(observations, Mapping):
        raise e9.BatchSizePilotError("Stage E.9 observations must be a mapping")
    e9.validate_observations(
        observations,
        project_root=project_root,
        verify_live_device=verify_live_device,
    )
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("results/stage_e9/htgnn_batch_size_pilot_observations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_e9/stage_e9_contract.json"),
    )
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()
    if not args.run_tests:
        parser.error("--run-tests is required when writing canonical E.9 evidence")
    root = args.project_root.resolve()
    observations_path = (
        args.observations
        if args.observations.is_absolute()
        else root / args.observations
    )
    output = args.output if args.output.is_absolute() else root / args.output
    observations = _load_validated_observations(
        observations_path,
        project_root=root,
        verify_live_device=True,
    )
    observations_physical_sha256 = _sha256(observations_path)
    frozen_hashes = e9.require_frozen_upstream(root)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    smoke = [
        _baseline_smoke(root, kind=kind, task=task, device_name=device)
        for device in devices
        for kind in baselines.BASELINE_KINDS
        for task in baselines.TASKS
    ]
    integrity_result = d4.verify_frozen_integrity(root)
    integrity = {
        "stage_c": {
            "status": "PASS"
            if integrity_result["stage_c_sha256_matches"]
            else "FAIL",
            "matched": integrity_result["stage_c_artifact_count"],
            "expected": integrity_result["stage_c_expected_count"],
        },
        "raw_csv": {
            "status": "PASS"
            if integrity_result["raw_csv_sha256_matches"]
            else "FAIL",
            "matched": integrity_result["raw_csv_count"],
            "expected": integrity_result["raw_csv_expected_count"],
        },
    }
    verification = {
        "focused_e9": _run_pytest(
            root,
            [
                "tests/test_stage_e9_baselines.py",
                "tests/test_stage_e9_batch_size_pilot.py",
                "tests/test_stage_e9_training_integration.py",
            ],
        ),
        "frozen_e2_e8_regression": _run_pytest(
            root,
            [
                "tests/test_pyg_adapter.py",
                "tests/test_e2_provenance_cache.py",
                "tests/test_feature_encoder.py",
                "tests/test_hierarchical_encoder.py",
                "tests/test_htgnn_classifier.py",
                "tests/test_resource_pilot.py",
                "tests/test_training_harness.py",
                "tests/test_optimization_pilot.py",
            ],
        ),
        "complete_project_suite": _run_pytest(root, []),
    }
    verification["status"] = (
        "PASS"
        if all(item["status"] == "PASS" for item in verification.values())
        else "FAIL"
    )
    source_hashes = {
        relative: _sha256(root / relative) for relative in E9_SOURCE_PATHS
    }
    all_smoke_passed = all(record["status"] == "PASS" for record in smoke)
    integrity_passed = all(item["status"] == "PASS" for item in integrity.values())
    report = {
        "stage": "E.9",
        "contract_commit": e9.CONTRACT_COMMIT,
        "baseline_version": baselines.BASELINE_VERSION,
        "batch_size_pilot_version": e9.PILOT_VERSION,
        "baseline_contract": {
            "final_flow": {
                "identity": baselines.FINAL_FLOW,
                "input": "float32[B,78] from temporal position five only",
                "architecture": [78, 64, 32],
                "parameter_counts": dict(
                    baselines.EXPECTED_PARAMETER_COUNTS[baselines.FINAL_FLOW]
                ),
            },
            "flattened_five_flow": {
                "identity": baselines.FLATTENED_FIVE_FLOW,
                "input": "float32[B,5,78] flattened temporal-major to [B,390]",
                "architecture": [390, 64, 32],
                "parameter_counts": dict(
                    baselines.EXPECTED_PARAMETER_COUNTS[
                        baselines.FLATTENED_FIVE_FLOW
                    ]
                ),
            },
            "task_heads": {"E1": 15, "E2": 15, "E3": 1},
            "output": "raw logits; E3 shape [B]",
            "graph_dependency": False,
            "attention_or_pruning": False,
            "state_dict_schema_sha256": {
                kind: {
                    task: e7.model_state_schema_sha256(
                        integration.BaselineTrainingAdapter(
                            baselines.create_baseline(kind, task),
                            kind=kind,
                            task=task,
                        )
                    )
                    for task in baselines.TASKS
                }
                for kind in baselines.BASELINE_KINDS
            },
        },
        "training_contract_reuse": {
            "loss_factory": "src.training.harness.create_loss",
            "optimizer_factory": "src.training.harness.create_optimizer",
            "train_epoch": "src.training.harness.train_one_epoch",
            "validation_epoch": "src.training.harness.validate_one_epoch",
            "early_stopping": "src.training.harness.EarlyStopping",
            "authoritative_batching": "D4 -> D6 -> E.2",
            "new_training_implementation": False,
            "checkpoint_identity": "E.7 model_state_schema_sha256 reused",
            "log_ownership": "E.7 exclusive reservation and alias rejection reused",
            "test_accessed": False,
        },
        "baseline_smoke": smoke,
        "batch_size_pilot": {
            "contract": e9.pilot_contract(),
            "observations_path": observations_path.relative_to(root).as_posix(),
            "observations_physical_sha256": observations_physical_sha256,
            "observation_file_hash_bound": True,
            "production_compatibility": observations["production_compatibility"],
            "all_candidates_complete": all(
                record["status"] == "COMPLETE"
                for record in observations["candidate_results"]
            ),
            "all_candidates_resource_eligible": all(
                record["resource_eligible"] is True
                for record in observations["candidate_results"]
            ),
            "selected_htgnn_training_batch_size": observations["selection"][
                "selected_batch_size"
            ],
            "selection_reproduced": True,
            "noisy_values_canonicalized": False,
        },
        "finalized": {
            "htgnn_training_batch_size": observations["selection"][
                "selected_batch_size"
            ]
        },
        "not_finalized": observations["not_finalized"],
        "frozen_upstream_source_sha256": frozen_hashes,
        "e9_source_sha256": source_hashes,
        "integrity": integrity,
        "verification": verification,
        "e10_implemented": False,
        "overall_status": (
            "PASS"
            if all_smoke_passed
            and integrity_passed
            and verification["status"] == "PASS"
            else "FAIL"
        ),
    }
    _write_contract(report, output)
    e9.validate_observation_file(
        observations_path,
        project_root=root,
        expected_sha256=report["batch_size_pilot"][
            "observations_physical_sha256"
        ],
        verify_live_device=True,
    )
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"physical_sha256={_sha256(output)}")
    print(f"selected_htgnn_training_batch_size={report['finalized']['htgnn_training_batch_size']}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
