"""Validate the authorized E.2 provenance-cache correction and write evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import batching as d6  # noqa: E402
from src.graph import pyg_adapter as e2  # noqa: E402
from src.graph import resource_pilot as e6  # noqa: E402
from src.graph import schema as d2  # noqa: E402


CORRECTION_VERSION = "stage-e2-provenance-cache-v1"
BASELINE_COMMIT = "67889f3315bcb227b89c3c3e929afcde534ca800"
BASELINE_ADAPTER_SHA256 = (
    "b0236db682d60d83a9f0d69f9a72ce50a2f8c7e73887c98637d6168d84782404"
)
BASELINE_FINGERPRINTS = {
    ("E1", "train"): "78ddb7ee851030b8402782b3f8e4aaf3999e42ab02d3fa3585ceeab310fc9135",
    ("E1", "validation"): "cc35a7fc7fbe19b83f77ec9ffd95c59b4653e0a0841f91412bb8a263a98fb3de",
    ("E1", "test"): "6c7d7498adff9fd03924f23882c583ef0f572f52a62521695260f5cc6c9a7232",
    ("E2", "train"): "bb50cddec01eb3e234cdab21f263d56557887695448b924babd83735717ec0d4",
    ("E2", "validation"): "74078a5ac4f6b6da456ff97c0577ac68ceb5cc0aa83a2bd427ea4d32daecd142",
    ("E2", "test"): "a1938b65ea97ce919bc6c9f98b3f602edd53f8615c28edea7af0368b5dc2dee9",
    ("E3", "train"): "dc58ad9926a861441deaf5e88270c2b874e609b71a64a6f7ed7fe510ebabbce2",
    ("E3", "validation"): "98b046bc430900fbd38b855ef5145526b0e733657dc70a48f990cd84d187034b",
    ("E3", "test"): "01690c32631e2b822a7ba11e02c21242b33e7c0d65cc86ff51e3bd9a5116a469",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_fingerprint(data: Any) -> str:
    digest = hashlib.sha256()
    for node_type in data.node_types:
        digest.update(node_type.encode("utf-8"))
        for key, value in sorted(data[node_type].items()):
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(
                    json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
                )
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    for edge_type in data.edge_types:
        digest.update(repr(edge_type).encode("utf-8"))
        for key, value in sorted(data[edge_type].items()):
            digest.update(key.encode("utf-8"))
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(
                    json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
                )
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    digest.update(
        json.dumps(data.e2_audit, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _measure_ms(operation: Any) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = operation()
    return result, round((time.perf_counter_ns() - start) / 1_000_000.0, 6)


def _timing_summary(samples: list[float]) -> dict[str, Any]:
    return e6.summarize_timings_ms(samples)


def _write_canonical_report(report: dict[str, Any], output: Path) -> None:
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


def _write_observations(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_validation(
    project_root: Path, measured_iterations: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = Path(project_root).resolve()
    if measured_iterations < 2 or measured_iterations > e6.MAX_MEASURED_ITERATIONS:
        raise ValueError(
            f"measured_iterations must be in [2, {e6.MAX_MEASURED_ITERATIONS}]"
        )
    frozen_observations_path = (
        project_root
        / "results"
        / "stage_e6"
        / "htgnn_resource_pilot_observations.json"
    )
    frozen_observations = json.loads(
        frozen_observations_path.read_text(encoding="utf-8")
    )
    source_paths = {
        "d8_report": project_root
        / "results"
        / "stage_d8"
        / "full_split_integration.json",
        "e1_environment_report": project_root
        / "results"
        / "stage_e1"
        / "environment-report.json",
    }
    source_before = {
        name: {"sha256": _sha256(path), "size": path.stat().st_size}
        for name, path in source_paths.items()
    }

    pair_records: list[dict[str, Any]] = []
    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(experiment, split)
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            batch = d6.build_logical_batch(dataset, indices)
            e2._reset_frozen_identity_cache()
            cold = e2.logical_batch_to_heterodata(batch, project_root=project_root)
            warm = e2.logical_batch_to_heterodata(batch, project_root=project_root)
            cold_fingerprint = _output_fingerprint(cold)
            warm_fingerprint = _output_fingerprint(warm)
            baseline = BASELINE_FINGERPRINTS[(experiment, split)]
            pair_records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "sample_indices": indices,
                    "baseline_fingerprint_sha256": baseline,
                    "cold_fingerprint_sha256": cold_fingerprint,
                    "warm_fingerprint_sha256": warm_fingerprint,
                    "baseline_cold_warm_equivalent": (
                        baseline == cold_fingerprint == warm_fingerprint
                    ),
                }
            )

    post_conversion: list[dict[str, Any]] = []
    for task in e6.TASKS:
        dataset = d4.LazyGraphDataset(task, "train")
        indices = e6.bounded_sample_indices(len(dataset), e6.DEFAULT_BATCH_SIZE)
        batch = d6.build_logical_batch(dataset, indices)
        e2._reset_frozen_identity_cache()
        cold_data, cold_ms = _measure_ms(
            lambda: e2.logical_batch_to_heterodata(batch, project_root=project_root)
        )
        warm_samples: list[float] = []
        warm_fingerprints: set[str] = set()
        for _ in range(measured_iterations):
            warm_data, elapsed_ms = _measure_ms(
                lambda: e2.logical_batch_to_heterodata(
                    batch, project_root=project_root
                )
            )
            warm_samples.append(elapsed_ms)
            warm_fingerprints.add(_output_fingerprint(warm_data))
        post_conversion.append(
            {
                "task": task,
                "batch_size": batch.batch_size,
                "cold_conversion_ms": cold_ms,
                "warm_conversion": _timing_summary(warm_samples),
                "cold_warm_output_identical": warm_fingerprints
                == {_output_fingerprint(cold_data)},
            }
        )

    pilot_config = e6.PilotConfig(
        batch_size=e6.DEFAULT_BATCH_SIZE,
        warmup_iterations=e6.DEFAULT_WARMUP_ITERATIONS,
        measured_iterations=measured_iterations,
    )
    e6_contract, e6_observations = e6.run_real_pilot(project_root, pilot_config)
    integrity = d4.verify_frozen_integrity(project_root)
    source_after = {
        name: {"sha256": _sha256(path), "size": path.stat().st_size}
        for name, path in source_paths.items()
    }
    semantic_pass = all(
        item["baseline_cold_warm_equivalent"] for item in pair_records
    )
    e6_pass = all(
        item["forward_contract"]["status"] == "PASS"
        and item["parameter_count"] == item["expected_parameter_count"]
        and item["batch_size"] == e6.DEFAULT_BATCH_SIZE
        for item in e6_contract["tasks"]
    )
    integrity_pass = bool(
        integrity["stage_c_sha256_matches"] and integrity["raw_csv_sha256_matches"]
    )
    deterministic = {
        "stage": "E.2 provenance-validation performance correction",
        "correction_version": CORRECTION_VERSION,
        "frozen_semantic_oracle": {
            "commit": BASELINE_COMMIT,
            "adapter_physical_sha256": BASELINE_ADAPTER_SHA256,
        },
        "cache_contract": {
            "scope": "compact frozen provenance identity strings only",
            "maximum_entries": 1,
            "first_call_requires_full_validation": True,
            "normal_write_or_replacement_invalidates": True,
            "key_fields": [
                "resolved_project_root",
                "approved_provenance_constants",
                "approved_report_hashes",
                "resolved_artifact_paths",
                "device_and_file_identity",
                "mode_and_link_count",
                "size",
                "mtime_ns",
                "ctime_ns",
            ],
            "validation_race_rejected": True,
            "cached_values_immutable": True,
            "fresh_identity_dictionary_per_call": True,
            "logical_batches_or_tensors_cached": False,
        },
        "semantic_equivalence": {
            "pair_count": len(pair_records),
            "pairs": pair_records,
            "all_pairs_equivalent": semantic_pass,
        },
        "e6_regression": {
            "batch_size": e6.DEFAULT_BATCH_SIZE,
            "tasks": [
                {
                    "task": item["task"],
                    "parameter_count": item["parameter_count"],
                    "expected_parameter_count": item["expected_parameter_count"],
                    "logit_shape": item["forward_contract"]["logit_shape"],
                    "feature_attention_shape": item["forward_contract"][
                        "feature_attention_shape"
                    ],
                    "category_attention_shape": item["forward_contract"][
                        "category_attention_shape"
                    ],
                    "forward_status": item["forward_contract"]["status"],
                }
                for item in e6_contract["tasks"]
            ],
            "all_contracts_pass": e6_pass,
            "runtime_measurements_are_noncanonical": True,
        },
        "frozen_provenance_sources": {
            "before": source_before,
            "after": source_after,
            "unchanged": source_before == source_after,
        },
        "integrity": {
            "stage_c_artifacts": {
                "matched": integrity["stage_c_artifact_count"],
                "expected": integrity["stage_c_expected_count"],
                "status": "PASS" if integrity["stage_c_sha256_matches"] else "FAIL",
            },
            "raw_csv_files": {
                "matched": integrity["raw_csv_count"],
                "expected": integrity["raw_csv_expected_count"],
                "status": "PASS" if integrity["raw_csv_sha256_matches"] else "FAIL",
            },
        },
        "excluded": {
            "training": True,
            "pruning": True,
            "predictive_metrics": True,
            "e7": True,
        },
        "overall_status": (
            "PASS"
            if semantic_pass
            and e6_pass
            and integrity_pass
            and source_before == source_after
            else "FAIL"
        ),
    }
    prior_by_task = {item["task"]: item for item in frozen_observations["tasks"]}
    post_e6_by_task = {item["task"]: item for item in e6_observations["tasks"]}
    observations = {
        "stage": "E.2 provenance-validation performance correction",
        "correction_version": CORRECTION_VERSION,
        "evidence_class": "runtime_performance_observations",
        "canonical_or_byte_reproducible": False,
        "hardware_specific": True,
        "measurement_note": (
            "Before values are from the frozen E.6 run and after values are from "
            "this correction run; normal run-to-run noise applies."
        ),
        "tasks": [
            {
                "task": task,
                "before_frozen_e6": {
                    "conversion_first_call_ms": prior_by_task[task][
                        "e2_conversion"
                    ]["first_call_ms"],
                    "conversion_repeated": prior_by_task[task]["e2_conversion"][
                        "repeated_calls"
                    ],
                    "host_to_device": prior_by_task[task]["host_to_device"][
                        "timing"
                    ],
                    "normal_forward": prior_by_task[task]["normal_forward"][
                        "timing"
                    ],
                    "attention_forward": prior_by_task[task]["attention_forward"][
                        "timing"
                    ],
                },
                "after_correction": {
                    "conversion_cold_call_ms": next(
                        item["cold_conversion_ms"]
                        for item in post_conversion
                        if item["task"] == task
                    ),
                    "conversion_warm": next(
                        item["warm_conversion"]
                        for item in post_conversion
                        if item["task"] == task
                    ),
                    "cold_warm_output_identical": next(
                        item["cold_warm_output_identical"]
                        for item in post_conversion
                        if item["task"] == task
                    ),
                    "host_to_device": post_e6_by_task[task]["host_to_device"][
                        "timing"
                    ],
                    "normal_forward": post_e6_by_task[task]["normal_forward"][
                        "timing"
                    ],
                    "attention_forward": post_e6_by_task[task][
                        "attention_forward"
                    ]["timing"],
                },
            }
            for task in e6.TASKS
        ],
        "no_arbitrary_improvement_threshold_applied": True,
        "h2d_or_forward_optimization_performed": False,
    }
    return deterministic, observations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the bounded E.2 provenance-cache correction"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--measured-iterations", type=int, default=e6.DEFAULT_MEASURED_ITERATIONS
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = project_root / "results" / "stage_e2"
    canonical_path = output_dir / "provenance_cache_validation.json"
    observations_path = output_dir / "provenance_cache_observations.json"
    deterministic, observations = run_validation(
        project_root, args.measured_iterations
    )
    _write_observations(observations, observations_path)
    _write_canonical_report(deterministic, canonical_path)
    print(f"overall_status={deterministic['overall_status']}")
    print(f"report_sha256={deterministic['report_sha256']}")
    print(f"physical_sha256={_sha256(canonical_path)}")
    print(f"canonical_output={canonical_path}")
    print(f"observations_output={observations_path}")
    return 0 if deterministic["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
