from __future__ import annotations

import copy

import pytest

from src.training import batch_size_pilot as e9


def _record(batch_size: int, throughput: float, *, eligible: bool = True):
    peak_fraction = 0.5 if eligible else 0.85
    total_vram = 1_000
    peak_reserved = int(peak_fraction * total_vram)
    return {
        "batch_size": batch_size,
        "status": "COMPLETE",
        "resource_eligible": eligible,
        "warmup_steps_completed": e9.WARMUP_STEPS,
        "measured_steps_completed": e9.MEASURED_STEPS,
        "losses_finite": True,
        "gradients_finite": True,
        "optimizer_steps_succeeded": True,
        "median_examples_per_second": throughput,
        "examples_per_second_samples": [throughput] * e9.MEASURED_STEPS,
        "latency_ms_median": 1.0,
        "latency_ms_p95_nearest_rank": 1.0,
        "peak_reserved_vram_fraction": peak_fraction,
        "baseline_allocated_bytes": 10,
        "baseline_reserved_bytes": 20,
        "peak_allocated_bytes": 30,
        "peak_reserved_bytes": peak_reserved,
        "total_vram_bytes": total_vram,
        "first_32_source_indices_sha256": "a" * 64,
        "latency_ms_samples": [1.0] * e9.MEASURED_STEPS,
    }


def _observations(records):
    return {
        "stage": "E.9",
        "pilot_version": e9.PILOT_VERSION,
        "execution_id": "e9-20260928t000000z",
        "executed_at_utc": "2026-09-28T00:00:00+00:00",
        "executor": "test",
        "elapsed_seconds": 1.0,
        "contract_commit": e9.CONTRACT_COMMIT,
        "contract": e9.pilot_contract(),
        "e9_source_sha256": {
            "src/training/batch_size_pilot.py": "a" * 64,
            "scripts/run_stage_e9_batch_size_pilot.py": "b" * 64,
        },
        "device": {"type": "cuda"},
        "candidate_results": records,
        "selection": e9.select_batch_size(records),
        "validation_accessed": False,
        "test_accessed": False,
        "pilot_weights_preserved": False,
        "overall_status": "PASS",
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
        _record(32, 100.0),
        _record(64, 104.0),
        _record(128, 105.0),
        _record(256, 200.0, eligible=False),
    ]
    selection = e9.select_batch_size(records)
    assert selection["maximum_median_examples_per_second"] == 105.0
    assert selection["throughput_tied_batch_sizes"] == [32, 64, 128]
    assert selection["selected_batch_size"] == 32


def test_selection_rejects_missing_reordered_and_no_eligible_candidates():
    records = [_record(size, float(size)) for size in e9.BATCH_SIZE_CANDIDATES]
    with pytest.raises(e9.BatchSizePilotError, match="missing, duplicated, or reordered"):
        e9.select_batch_size(list(reversed(records)))
    for record in records:
        record["resource_eligible"] = False
    with pytest.raises(e9.BatchSizePilotError, match="no batch-size"):
        e9.select_batch_size(records)


def test_strict_vram_threshold_is_revalidated():
    records = [_record(size, float(size)) for size in e9.BATCH_SIZE_CANDIDATES]
    report = _observations(records)
    corrupted = copy.deepcopy(report)
    corrupted["candidate_results"][0]["peak_reserved_bytes"] = 800
    corrupted["candidate_results"][0]["peak_reserved_vram_fraction"] = 0.80
    with pytest.raises(e9.BatchSizePilotError, match="eligibility"):
        e9.validate_observations(corrupted)


def test_observations_reproduce_selection_and_require_all_samples():
    records = [_record(size, float(size)) for size in e9.BATCH_SIZE_CANDIDATES]
    report = _observations(records)
    e9.validate_observations(report)
    corrupted = copy.deepcopy(report)
    corrupted["selection"]["selected_batch_size"] = 32
    with pytest.raises(e9.BatchSizePilotError, match="not reproducible"):
        e9.validate_observations(corrupted)
    corrupted = copy.deepcopy(report)
    corrupted["candidate_results"][0]["latency_ms_samples"].pop()
    with pytest.raises(e9.BatchSizePilotError, match="timing samples"):
        e9.validate_observations(corrupted)


def test_nearest_rank_p95_retains_individual_observations():
    values = [float(value) for value in range(1, 21)]
    assert e9._nearest_rank_p95(values) == 19.0
