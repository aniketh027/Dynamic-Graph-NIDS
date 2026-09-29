from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from src.graph import adapter as d4
from scripts import validate_stage_e9 as evidence
from src.training import batch_size_pilot as e9


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(batch_size: int, *, eligible: bool = True) -> dict:
    latency = [float(index + 1) for index in range(e9.MEASURED_STEPS)]
    throughput = [batch_size * 1000.0 / value for value in latency]
    peak_reserved = 40 if eligible else int(0.85 * e9.EXPECTED_TOTAL_VRAM_BYTES)
    steps = [
        {
            "phase": "warmup",
            "step_index": index,
            "completed": True,
            "loss_value": 1.0,
            "loss_finite": True,
            "gradients_finite": True,
            "optimizer_step_succeeded": True,
            "cuda_oom": False,
            "failure_stage": None,
            "latency_ms": None,
        }
        for index in range(1, e9.WARMUP_STEPS + 1)
    ]
    steps.extend(
        {
            "phase": "measured",
            "step_index": index,
            "completed": True,
            "loss_value": 1.0,
            "loss_finite": True,
            "gradients_finite": True,
            "optimizer_step_succeeded": True,
            "cuda_oom": False,
            "failure_stage": None,
            "latency_ms": value,
        }
        for index, value in enumerate(latency, start=1)
    )
    return {
        "batch_size": batch_size,
        "execution_id": "e9-20260929t000000z",
        "config_identity_sha256": e9._config_identity_sha256(),
        "status": e9.COMPLETE_STATUS,
        "resource_eligible": eligible,
        "failure_type": None,
        "failure_message": None,
        "failure_stage": None,
        "warmup_steps_attempted": e9.WARMUP_STEPS,
        "warmup_steps_completed": e9.WARMUP_STEPS,
        "measured_steps_attempted": e9.MEASURED_STEPS,
        "measured_steps_completed": e9.MEASURED_STEPS,
        "step_results": steps,
        "losses_finite": True,
        "gradients_finite": True,
        "optimizer_steps_succeeded": True,
        "parameter_count": 7_183,
        "first_32_source_indices_sha256": "a" * 64,
        "all_candidate_source_indices_sha256": f"{batch_size:064x}",
        "timing_scope": "host_to_device_forward_loss_backward_optimizer_step",
        "cpu_d4_d6_e2_construction_timed": False,
        "warmups_excluded_from_timing": True,
        "latency_ms_samples": latency,
        "latency_ms_median": 10.5,
        "latency_ms_p95_nearest_rank": 19.0,
        "examples_per_second_samples": throughput,
        "median_examples_per_second": (throughput[9] + throughput[10]) / 2,
        "measured_loss_samples": [1.0] * e9.MEASURED_STEPS,
        "baseline_allocated_bytes": 10,
        "baseline_reserved_bytes": 20,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": 20,
        "peak_reserved_delta_bytes": peak_reserved - 20,
        "total_vram_bytes": e9.EXPECTED_TOTAL_VRAM_BYTES,
        "peak_reserved_vram_fraction": peak_reserved / e9.EXPECTED_TOTAL_VRAM_BYTES,
        "peak_reserved_vram_percent": (
            100.0 * peak_reserved / e9.EXPECTED_TOTAL_VRAM_BYTES
        ),
        "headroom_threshold_fraction": e9.HEADROOM_THRESHOLD,
    }


def _observations(records: list[dict]) -> dict:
    selection = e9.select_batch_size(records)
    selected = selection["selected_batch_size"]
    return {
        "stage": "E.9",
        "pilot_version": e9.PILOT_VERSION,
        "execution_id": "e9-20260929t000000z",
        "executed_at_utc": "2026-09-29T00:00:00+00:00",
        "executor": "test",
        "elapsed_seconds": 1.0,
        "contract_commit": e9.CONTRACT_COMMIT,
        "contract_source_sha256": {
            "PROJECT_HANDOFF.md@contract_commit": e9.CONTRACT_HANDOFF_SHA256
        },
        "contract": e9.pilot_contract(),
        "e9_source_sha256": {
            relative: "b" * 64 for relative in e9.OBSERVATION_SOURCE_PATHS
        },
        "frozen_upstream_source_sha256": {"frozen": "c" * 64},
        "device": {
            "type": "cuda",
            "name": e9.EXPECTED_DEVICE_NAME,
            "total_vram_bytes": e9.EXPECTED_TOTAL_VRAM_BYTES,
            "torch": "test",
            "cuda_runtime": "test",
        },
        "candidate_results": records,
        "selection": selection,
        "production_compatibility": {
            "status": "PASS",
            "task": "E2",
            "batch_size": selected,
            "training_config_valid": True,
            "authoritative_iterator": "src.training.harness.iter_production_batches",
            "training_helper": "src.training.harness.train_one_epoch",
            "logits_shape": [selected, 15],
            "train_batches": 1,
            "train_examples": selected,
            "loss_finite": True,
            "gradients_finite": True,
            "optimizer_step_succeeded": True,
            "validation_accessed": False,
            "test_accessed": False,
        },
        "finalized": {"htgnn_training_batch_size": selected},
        "not_finalized": [
            "learning_rate",
            "weighted_loss_configuration",
            "task_performance",
            "final_model_checkpoint",
            "test_performance",
        ],
        "validation_accessed": False,
        "test_accessed": False,
        "pilot_weights_preserved": False,
        "overall_status": "PASS",
    }


def _good_report() -> dict:
    return _observations([_record(size) for size in e9.BATCH_SIZE_CANDIDATES])


def _rebind_selection(report: dict) -> None:
    selection = e9.select_batch_size(report["candidate_results"])
    selected = selection["selected_batch_size"]
    report["selection"] = selection
    report["finalized"] = {"htgnn_training_batch_size": selected}
    compatibility = report["production_compatibility"]
    compatibility["batch_size"] = selected
    compatibility["logits_shape"] = [selected, 15]
    compatibility["train_examples"] = selected


def _honest_oom_record(batch_size: int) -> dict:
    complete = _record(batch_size)
    steps = complete["step_results"][: e9.WARMUP_STEPS + 3]
    failed = steps[-1]
    failed.update(
        {
            "completed": False,
            "loss_value": None,
            "loss_finite": None,
            "gradients_finite": None,
            "optimizer_step_succeeded": False,
            "cuda_oom": True,
            "failure_stage": "forward",
            "latency_ms": None,
        }
    )
    completed_measured = [
        step
        for step in steps
        if step["completed"] and step["phase"] == "measured"
    ]
    latency = [step["latency_ms"] for step in completed_measured]
    losses = [step["loss_value"] for step in completed_measured]
    return {
        "batch_size": batch_size,
        "execution_id": complete["execution_id"],
        "config_identity_sha256": complete["config_identity_sha256"],
        "status": e9.OOM_STATUS,
        "resource_eligible": False,
        "failure_type": "OutOfMemoryError",
        "failure_message": "synthetic honest CUDA OOM",
        "failure_stage": "forward",
        "warmup_steps_attempted": e9.WARMUP_STEPS,
        "warmup_steps_completed": e9.WARMUP_STEPS,
        "measured_steps_attempted": 3,
        "measured_steps_completed": 2,
        "step_results": steps,
        "losses_finite": False,
        "gradients_finite": False,
        "optimizer_steps_succeeded": False,
        "latency_ms_samples": latency,
        "examples_per_second_samples": [
            batch_size * 1000.0 / value for value in latency
        ],
        "measured_loss_samples": losses,
        "total_vram_bytes": e9.EXPECTED_TOTAL_VRAM_BYTES,
    }


def test_frozen_candidate_and_measurement_contract():
    contract = e9.pilot_contract()
    assert e9.BATCH_SIZE_CANDIDATES == (32, 64, 128, 256)
    assert contract["task"] == "E2"
    assert contract["seed"] == 2025
    assert contract["learning_rate"] == 3e-4
    assert contract["learning_rate_scope"] == "resource_measurement_only_not_finalized"
    assert contract["warmup_optimizer_steps"] == 5
    assert contract["measured_optimizer_steps"] == 20
    assert contract["data_partition"] == "E2 train only"
    assert contract["validation_accessed"] is False
    assert contract["test_accessed"] is False
    assert contract["absolute_peak_reserved_vram_threshold_fraction"] == 0.80


def test_each_candidate_restarts_the_same_seeded_traversal_prefix():
    batches = {
        size: list(e9.iter_pilot_index_batches(100_000, size))
        for size in e9.BATCH_SIZE_CANDIDATES
    }
    assert all(len(value) == e9.TOTAL_STEPS for value in batches.values())
    prefix = batches[32][0]
    assert all(value[0][:32] == prefix for value in batches.values())
    assert all(len(value[0]) == size for size, value in batches.items())
    assert len({index for group in batches[256] for index in group}) == (
        256 * e9.TOTAL_STEPS
    )


def test_traversal_rejects_unapproved_sizes_and_unbounded_sources():
    with pytest.raises(ValueError, match="batch_size"):
        list(e9.iter_pilot_index_batches(10_000, 512))
    with pytest.raises(e9.BatchSizePilotError, match="required"):
        list(e9.iter_pilot_index_batches(100, 256))


def test_selection_uses_eligibility_95_percent_tie_and_smallest_candidate():
    records = [
        {"batch_size": 32, "resource_eligible": True, "median_examples_per_second": 100.0},
        {"batch_size": 64, "resource_eligible": True, "median_examples_per_second": 104.0},
        {"batch_size": 128, "resource_eligible": True, "median_examples_per_second": 105.0},
        {"batch_size": 256, "resource_eligible": False, "median_examples_per_second": 200.0},
    ]
    selection = e9.select_batch_size(records)
    assert selection["throughput_tied_batch_sizes"] == [32, 64, 128]
    assert selection["selected_batch_size"] == 32


def test_selection_rejects_missing_reordered_and_no_eligible_candidates():
    records = [_record(size) for size in e9.BATCH_SIZE_CANDIDATES]
    with pytest.raises(e9.BatchSizePilotError, match="missing, duplicated, or reordered"):
        e9.select_batch_size(list(reversed(records)))
    for record in records:
        record["resource_eligible"] = False
    with pytest.raises(e9.BatchSizePilotError, match="no batch-size"):
        e9.select_batch_size(records)


def test_known_good_observation_fixture_passes_real_validator():
    e9.validate_observations(_good_report())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["device"].__setitem__("name", "wrong gpu"), "device"),
        (lambda r: r["device"].__setitem__("total_vram_bytes", 1), "device"),
        (lambda r: r["candidate_results"][0]["latency_ms_samples"].__setitem__(0, 2.0), "summaries"),
        (lambda r: r["candidate_results"][0].__setitem__("median_examples_per_second", 1.0), "summary"),
        (lambda r: r["candidate_results"][0].__setitem__("latency_ms_p95_nearest_rank", 1.0), "summary"),
        (lambda r: r["candidate_results"][0].__setitem__("peak_reserved_bytes", 41), "relationships|percentage"),
        (lambda r: r["candidate_results"][0].__setitem__("peak_reserved_vram_fraction", 0.5), "percentage"),
        (lambda r: r["candidate_results"][0].__setitem__("peak_reserved_vram_percent", 0.5), "percentage"),
        (lambda r: r["candidate_results"][0].__setitem__("losses_finite", False), "summaries"),
        (lambda r: r["candidate_results"][0].__setitem__("resource_eligible", False), "eligibility|selection"),
        (lambda r: r["candidate_results"][0].__setitem__("batch_size", 33), "candidate"),
        (lambda r: r["candidate_results"].pop(), "coverage"),
        (lambda r: r["selection"].__setitem__("selected_batch_size", 32), "selection"),
        (lambda r: r["contract"].__setitem__("seed", 2026), "contract"),
        (lambda r: r["contract"].__setitem__("learning_rate", 1e-3), "contract"),
        (lambda r: r["contract"].__setitem__("task", "E1"), "contract"),
        (lambda r: r["candidate_results"][0].__setitem__("warmup_steps_completed", 4), "summaries"),
        (lambda r: r.__setitem__("execution_id", "tampered"), "execution"),
        (lambda r: r["candidate_results"][0].__setitem__("config_identity_sha256", "d" * 64), "config"),
        (lambda r: r["contract_source_sha256"].__setitem__("PROJECT_HANDOFF.md@contract_commit", "d" * 64), "contract source"),
    ],
)
def test_adversarial_observation_mutations_fail_closed(mutation, message):
    corrupted = copy.deepcopy(_good_report())
    mutation(corrupted)
    with pytest.raises(e9.BatchSizePilotError, match=message):
        e9.validate_observations(corrupted)


def test_correlated_false_nonfinite_story_is_rejected_from_raw_trace():
    forged = _good_report()
    record = forged["candidate_results"][0]
    record.update(
        {
            "status": e9.NONFINITE_LOSS_STATUS,
            "resource_eligible": False,
            "failure_type": "SyntheticNonfinite",
            "failure_message": "forged summary",
            "failure_stage": "loss",
            "measured_steps_completed": 0,
            "losses_finite": False,
            "gradients_finite": False,
            "optimizer_steps_succeeded": False,
            "latency_ms_samples": [],
            "examples_per_second_samples": [],
            "measured_loss_samples": [],
        }
    )
    _rebind_selection(forged)
    with pytest.raises(e9.BatchSizePilotError, match="failed step|raw execution facts"):
        e9.validate_observations(forged)


def test_correlated_false_cuda_oom_story_is_rejected_from_raw_trace():
    forged = _good_report()
    record = forged["candidate_results"][0]
    record.update(
        {
            "status": e9.OOM_STATUS,
            "resource_eligible": False,
            "failure_type": "OutOfMemoryError",
            "failure_message": "forged OOM",
            "failure_stage": "forward",
            "warmup_steps_completed": 0,
            "measured_steps_completed": 0,
            "losses_finite": False,
            "gradients_finite": False,
            "optimizer_steps_succeeded": False,
        }
    )
    _rebind_selection(forged)
    with pytest.raises(e9.BatchSizePilotError, match="raw execution facts"):
        e9.validate_observations(forged)


def test_false_success_with_incomplete_raw_trace_is_rejected():
    forged = _good_report()
    forged["candidate_results"][0]["step_results"].pop()
    with pytest.raises(e9.BatchSizePilotError, match="failed step|raw execution facts"):
        e9.validate_observations(forged)


def test_optimizer_failure_raw_fact_cannot_be_summarized_as_success():
    forged = _good_report()
    failed = forged["candidate_results"][0]["step_results"][-1]
    failed.update(
        {
            "completed": False,
            "optimizer_step_succeeded": False,
            "failure_stage": "optimizer_step",
            "latency_ms": None,
        }
    )
    with pytest.raises(e9.BatchSizePilotError, match="raw execution facts"):
        e9.validate_observations(forged)


def test_raw_nonfinite_fact_cannot_be_summarized_as_success():
    forged = _good_report()
    failed = forged["candidate_results"][0]["step_results"][-1]
    failed.update(
        {
            "completed": False,
            "loss_value": None,
            "loss_finite": False,
            "gradients_finite": None,
            "optimizer_step_succeeded": False,
            "failure_stage": "loss",
            "latency_ms": None,
        }
    )
    with pytest.raises(e9.BatchSizePilotError, match="raw execution facts"):
        e9.validate_observations(forged)


def test_honest_partial_cuda_oom_candidate_is_valid_but_ineligible():
    records = [_record(size) for size in e9.BATCH_SIZE_CANDIDATES[:-1]]
    records.append(_honest_oom_record(e9.BATCH_SIZE_CANDIDATES[-1]))
    report = _observations(records)
    e9.validate_observations(report)
    assert report["candidate_results"][-1]["resource_eligible"] is False
    assert report["candidate_results"][-1]["status"] == e9.OOM_STATUS


def test_evidence_generator_refuses_correlated_forgery_before_binding(
    tmp_path: Path, monkeypatch
):
    forged = _good_report()
    record = forged["candidate_results"][0]
    record.update(
        {
            "status": e9.OOM_STATUS,
            "resource_eligible": False,
            "failure_type": "OutOfMemoryError",
            "failure_message": "forged OOM",
            "failure_stage": "forward",
            "warmup_steps_completed": 0,
            "measured_steps_completed": 0,
            "losses_finite": False,
            "gradients_finite": False,
            "optimizer_steps_succeeded": False,
        }
    )
    _rebind_selection(forged)
    monkeypatch.setattr(
        evidence.e9,
        "observation_source_sha256",
        lambda _root: dict(forged["e9_source_sha256"]),
    )
    monkeypatch.setattr(
        evidence.e9,
        "require_frozen_upstream",
        lambda _root: dict(forged["frozen_upstream_source_sha256"]),
    )
    validate_observations = evidence.e9.validate_observations

    def validate_without_live_device(report, *, project_root, verify_live_device):
        assert verify_live_device is True
        return validate_observations(
            report,
            project_root=project_root,
            verify_live_device=False,
        )

    monkeypatch.setattr(
        evidence.e9, "validate_observations", validate_without_live_device
    )
    observations = tmp_path / "forged.json"
    output = tmp_path / "must-not-exist.json"
    observations.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_stage_e9.py",
            "--project-root",
            str(Path(".").resolve()),
            "--observations",
            str(observations),
            "--output",
            str(output),
            "--run-tests",
        ],
    )
    with pytest.raises(e9.BatchSizePilotError, match="raw execution facts"):
        evidence.main()
    assert not output.exists()


def test_source_hash_corruption_fails_against_current_authority(monkeypatch):
    report = _good_report()
    expected_sources = dict(report["e9_source_sha256"])
    expected_upstream = dict(report["frozen_upstream_source_sha256"])
    monkeypatch.setattr(
        e9, "observation_source_sha256", lambda _root: expected_sources
    )
    monkeypatch.setattr(
        e9,
        "require_frozen_upstream",
        lambda _root: expected_upstream,
    )
    e9.validate_observations(report, project_root=Path("."))
    report["e9_source_sha256"][e9.OBSERVATION_SOURCE_PATHS[0]] = "d" * 64
    with pytest.raises(e9.BatchSizePilotError, match="source hash"):
        e9.validate_observations(report, project_root=Path("."))


def test_observation_file_hash_binding_detects_any_file_tampering(
    tmp_path: Path, monkeypatch
):
    report = _good_report()
    monkeypatch.setattr(
        e9, "observation_source_sha256", lambda _root: dict(report["e9_source_sha256"])
    )
    monkeypatch.setattr(
        e9,
        "require_frozen_upstream",
        lambda _root: dict(report["frozen_upstream_source_sha256"]),
    )
    path = tmp_path / "observations.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    e9.validate_observation_file(path, project_root=Path("."), expected_sha256=expected)
    corrupted = copy.deepcopy(report)
    corrupted["device"]["name"] = "tampered"
    path.write_text(json.dumps(corrupted, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(e9.BatchSizePilotError, match="SHA-256 binding"):
        e9.validate_observation_file(path, project_root=Path("."), expected_sha256=expected)


def test_nearest_rank_p95_retains_individual_observations():
    values = [float(value) for value in range(1, 21)]
    assert e9._nearest_rank_p95(values) == 19.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA replay requires CUDA")
def test_run_candidate_short_real_cuda_timing_and_memory_path():
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    dataset = d4.LazyGraphDataset(
        "E2",
        "train",
        processed_root=PROJECT_ROOT / "data" / "processed" / "preprocessed",
    )
    record = e9.run_candidate(
        dataset=dataset,
        batch_size=32,
        device=device,
        total_vram_bytes=int(properties.total_memory),
        execution_id="e9-20260929t000000z",
    )
    assert record["status"] == "COMPLETE"
    assert record["measured_steps_completed"] == e9.MEASURED_STEPS
    assert len(record["latency_ms_samples"]) == e9.MEASURED_STEPS
    assert record["peak_reserved_bytes"] >= record["baseline_reserved_bytes"]
    assert record["resource_eligible"] is True
