"""Stage E.9 bounded HTGNN E2 training batch-size resource pilot.

Only the authoritative E2 train stream is opened.  The module performs the
frozen five warmup plus twenty measured optimizer steps for candidates
32/64/128/256, records noisy CUDA observations separately, and applies the
frozen deterministic throughput/headroom selection rule.
"""

from __future__ import annotations

import gc
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Final, Iterator, Mapping, Sequence

import torch

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import pyg_adapter as e2
from src.graph import schema as d2
from src.training import harness as e7
from src.training import optimization_pilot as e8


PILOT_VERSION: Final = "stage-e9-htgnn-batch-size-pilot-v1"
TASK: Final = "E2"
SEED: Final = 2025
LEARNING_RATE: Final = 3e-4
BATCH_SIZE_CANDIDATES: Final = (32, 64, 128, 256)
WARMUP_STEPS: Final = 5
MEASURED_STEPS: Final = 20
TOTAL_STEPS: Final = WARMUP_STEPS + MEASURED_STEPS
HEADROOM_THRESHOLD: Final = 0.80
THROUGHPUT_TIE_FRACTION: Final = 0.95
CONTRACT_COMMIT: Final = "76d1bf81f2e18e3d0b99755a2d87bff9d96f3f0f"
CONTRACT_HANDOFF_SHA256: Final = (
    "ca0e9ba0d91d8997f7550a618f9cf3b670016b8bba719e70d5da5f293719fa54"
)
EXPECTED_DEVICE_NAME: Final = "NVIDIA GeForce RTX 4050 Laptop GPU"
EXPECTED_TOTAL_VRAM_BYTES: Final = 6_438_780_928
OBSERVATION_SOURCE_PATHS: Final = (
    "src/training/batch_size_pilot.py",
    "scripts/run_stage_e9_batch_size_pilot.py",
    "src/training/harness.py",
    "src/training/optimization_pilot.py",
    "src/training/baselines.py",
    "src/training/baseline_integration.py",
)
_EXECUTION_ID_PATTERN: Final = re.compile(r"^e9-\d{8}t\d{6}z$")
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class BatchSizePilotError(RuntimeError):
    """Raised when Stage E.9 resource evidence violates its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_source_sha256(project_root: Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    return {relative: _sha256(root / relative) for relative in OBSERVATION_SOURCE_PATHS}


def _config_identity_sha256() -> str:
    return d2.canonical_sha256(pilot_contract())


def pilot_contract() -> dict[str, Any]:
    return {
        "task": TASK,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "learning_rate_scope": "resource_measurement_only_not_finalized",
        "loss": "unweighted CrossEntropyLoss",
        "optimizer": "AdamW",
        "weight_decay": e7.WEIGHT_DECAY,
        "scheduler": None,
        "batch_size_candidates": list(BATCH_SIZE_CANDIDATES),
        "warmup_optimizer_steps": WARMUP_STEPS,
        "measured_optimizer_steps": MEASURED_STEPS,
        "data_partition": "E2 train only",
        "validation_accessed": False,
        "test_accessed": False,
        "absolute_peak_reserved_vram_threshold_fraction": HEADROOM_THRESHOLD,
        "throughput_tie_fraction": THROUGHPUT_TIE_FRACTION,
        "selection_rule": (
            "smallest candidate with median examples/sec at least 95% of the "
            "maximum among resource-eligible candidates"
        ),
    }


def require_frozen_upstream(project_root: Path) -> dict[str, str]:
    """Verify frozen E.2-E.8 sources before any resource measurement."""

    root = Path(project_root).resolve()
    hashes = e8.require_frozen_contracts(root)
    return hashes


def iter_pilot_index_batches(
    dataset_length: int, batch_size: int
) -> Iterator[tuple[int, ...]]:
    """Restart the frozen seeded affine traversal for one candidate."""

    if batch_size not in BATCH_SIZE_CANDIDATES:
        raise ValueError(f"batch_size must be one of {BATCH_SIZE_CANDIDATES}")
    if isinstance(dataset_length, bool) or not isinstance(dataset_length, int):
        raise TypeError("dataset_length must be an integer")
    required = batch_size * TOTAL_STEPS
    if dataset_length < required:
        raise BatchSizePilotError(
            f"E2 train stream has {dataset_length} rows; {required} are required"
        )
    multiplier, offset = e7._affine_permutation_parameters(dataset_length, SEED)
    for batch_number in range(TOTAL_STEPS):
        start = batch_number * batch_size
        yield tuple(
            (multiplier * position + offset) % dataset_length
            for position in range(start, start + batch_size)
        )


def _nearest_rank_p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _finite_parameters(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _training_step(
    *,
    data: Any,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, bool, bool]:
    """Execute one H2D -> forward -> loss -> backward -> optimizer step."""

    moved = data.to(device)
    targets = moved["sample"].y
    optimizer.zero_grad(set_to_none=True)
    logits = model(moved)
    if isinstance(logits, tuple):
        raise BatchSizePilotError("resource pilot must not collect attention")
    loss = e7.compute_loss(TASK, criterion, logits, targets)
    loss_finite = bool(torch.isfinite(loss))
    if not loss_finite:
        raise BatchSizePilotError("nonfinite resource-pilot loss")
    loss.backward()
    e7._finite_gradients(model)
    gradients_finite = True
    optimizer.step()
    parameters_finite = _finite_parameters(model)
    if not parameters_finite:
        raise BatchSizePilotError("optimizer produced nonfinite parameters")
    loss_value = float(loss.detach().cpu())
    del logits, loss, targets, moved
    return loss_value, gradients_finite, parameters_finite


def _candidate_failure_record(
    batch_size: int,
    *,
    execution_id: str,
    status: str,
    failure_type: str,
    failure_message: str,
    total_vram_bytes: int,
) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "execution_id": execution_id,
        "config_identity_sha256": _config_identity_sha256(),
        "status": status,
        "resource_eligible": False,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "warmup_steps_completed": 0,
        "measured_steps_completed": 0,
        "losses_finite": False,
        "gradients_finite": False,
        "optimizer_steps_succeeded": False,
        "total_vram_bytes": total_vram_bytes,
    }


def run_candidate(
    *,
    dataset: d4.LazyGraphDataset,
    batch_size: int,
    device: torch.device,
    total_vram_bytes: int,
    execution_id: str,
) -> dict[str, Any]:
    """Run one isolated candidate and return observational measurements."""

    if device.type != "cuda":
        raise BatchSizePilotError("Stage E.9 batch-size measurement requires CUDA")
    if dataset.experiment != TASK or dataset.split != "train":
        raise BatchSizePilotError("pilot dataset must be authoritative E2/train")
    if not _EXECUTION_ID_PATTERN.fullmatch(execution_id):
        raise BatchSizePilotError("candidate execution identity is invalid")
    e7.seed_everything(SEED)
    torch.cuda.empty_cache()
    model = e7.create_model(TASK).to(device).train()
    criterion = e7.create_loss(TASK).to(device)
    optimizer = e7.create_optimizer(model, LEARNING_RATE)
    batches = iter_pilot_index_batches(len(dataset), batch_size)
    first_prefix: tuple[int, ...] | None = None
    full_order = hashlib.sha256()
    warmup_completed = 0
    measured_completed = 0
    losses_finite = True
    gradients_finite = True
    optimizer_steps_succeeded = True
    timings_ms: list[float] = []
    throughput_samples: list[float] = []
    losses: list[float] = []

    try:
        for step_index, indices in enumerate(batches):
            if first_prefix is None:
                first_prefix = tuple(indices[: BATCH_SIZE_CANDIDATES[0]])
            for index in indices:
                full_order.update(int(index).to_bytes(8, "little", signed=False))
            logical = d6.build_logical_batch(dataset, indices)
            cpu_data = e2.logical_batch_to_heterodata(logical)
            if step_index < WARMUP_STEPS:
                loss_value, gradients_ok, parameters_ok = _training_step(
                    data=cpu_data,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                )
                torch.cuda.synchronize(device)
                warmup_completed += 1
                losses_finite = losses_finite and math.isfinite(loss_value)
                gradients_finite = gradients_finite and gradients_ok
                optimizer_steps_succeeded = optimizer_steps_succeeded and parameters_ok
                del cpu_data, logical
                continue

            if step_index == WARMUP_STEPS:
                torch.cuda.synchronize(device)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                baseline_allocated = int(torch.cuda.memory_allocated(device))
                baseline_reserved = int(torch.cuda.memory_reserved(device))

            torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss_value, gradients_ok, parameters_ok = _training_step(
                data=cpu_data,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms <= 0 or not math.isfinite(elapsed_ms):
                raise BatchSizePilotError("measured CUDA latency is invalid")
            timings_ms.append(elapsed_ms)
            throughput_samples.append(batch_size * 1000.0 / elapsed_ms)
            losses.append(loss_value)
            measured_completed += 1
            losses_finite = losses_finite and math.isfinite(loss_value)
            gradients_finite = gradients_finite and gradients_ok
            optimizer_steps_succeeded = optimizer_steps_succeeded and parameters_ok
            del cpu_data, logical

        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        peak_reserved_fraction = peak_reserved / total_vram_bytes
        eligible = bool(
            warmup_completed == WARMUP_STEPS
            and measured_completed == MEASURED_STEPS
            and losses_finite
            and gradients_finite
            and optimizer_steps_succeeded
            and peak_reserved_fraction < HEADROOM_THRESHOLD
        )
        assert first_prefix is not None
        prefix_digest = hashlib.sha256()
        for index in first_prefix:
            prefix_digest.update(int(index).to_bytes(8, "little", signed=False))
        return {
            "batch_size": batch_size,
            "execution_id": execution_id,
            "config_identity_sha256": _config_identity_sha256(),
            "status": "COMPLETE",
            "resource_eligible": eligible,
            "warmup_steps_completed": warmup_completed,
            "measured_steps_completed": measured_completed,
            "losses_finite": losses_finite,
            "gradients_finite": gradients_finite,
            "optimizer_steps_succeeded": optimizer_steps_succeeded,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "first_32_source_indices_sha256": prefix_digest.hexdigest(),
            "all_candidate_source_indices_sha256": full_order.hexdigest(),
            "timing_scope": "host_to_device_forward_loss_backward_optimizer_step",
            "cpu_d4_d6_e2_construction_timed": False,
            "warmups_excluded_from_timing": True,
            "latency_ms_samples": timings_ms,
            "latency_ms_median": statistics.median(timings_ms),
            "latency_ms_p95_nearest_rank": _nearest_rank_p95(timings_ms),
            "examples_per_second_samples": throughput_samples,
            "median_examples_per_second": statistics.median(throughput_samples),
            "measured_loss_samples": losses,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_delta_bytes": peak_allocated - baseline_allocated,
            "peak_reserved_delta_bytes": peak_reserved - baseline_reserved,
            "total_vram_bytes": total_vram_bytes,
            "peak_reserved_vram_fraction": peak_reserved_fraction,
            "peak_reserved_vram_percent": 100.0 * peak_reserved_fraction,
            "headroom_threshold_fraction": HEADROOM_THRESHOLD,
        }
    except torch.OutOfMemoryError as exc:
        return _candidate_failure_record(
            batch_size,
            execution_id=execution_id,
            status="CUDA_OOM",
            failure_type=type(exc).__name__,
            failure_message=str(exc),
            total_vram_bytes=total_vram_bytes,
        )
    finally:
        del optimizer, criterion, model
        gc.collect()
        torch.cuda.empty_cache()


def select_batch_size(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the exact frozen eligibility/95%-tie/smallest-candidate rule."""

    if [record.get("batch_size") for record in records] != list(
        BATCH_SIZE_CANDIDATES
    ):
        raise BatchSizePilotError("candidate records are missing, duplicated, or reordered")
    eligible = [record for record in records if record.get("resource_eligible") is True]
    if not eligible:
        raise BatchSizePilotError("no batch-size candidate is resource-eligible")
    throughputs = []
    for record in eligible:
        value = record.get("median_examples_per_second")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise BatchSizePilotError("eligible candidate has invalid median throughput")
        throughputs.append(float(value))
    maximum = max(throughputs)
    threshold = THROUGHPUT_TIE_FRACTION * maximum
    tied = sorted(
        int(record["batch_size"])
        for record in eligible
        if float(record["median_examples_per_second"]) >= threshold
    )
    return {
        "maximum_median_examples_per_second": maximum,
        "throughput_tie_threshold_examples_per_second": threshold,
        "throughput_tied_batch_sizes": tied,
        "selected_batch_size": tied[0],
        "rule": pilot_contract()["selection_rule"],
    }


def verify_production_compatibility(
    dataset: d4.LazyGraphDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Exercise one selected batch through the actual production train helper."""

    config = e7.TrainingConfig(
        task=TASK,
        learning_rate=LEARNING_RATE,
        seed=SEED,
        batch_size=batch_size,
        max_epochs=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    model = e7.create_model(TASK).to(device)
    shapes: list[tuple[int, ...]] = []
    hook = model.register_forward_hook(
        lambda _module, _inputs, output: shapes.append(tuple(output.shape))
    )
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    try:
        stats = e7.train_one_epoch(
            task=TASK,
            model=model,
            batches=e7.iter_production_batches(
                dataset,
                task=TASK,
                split="train",
                batch_size=config.batch_size,
                epoch_seed=config.seed,
                max_batches=1,
            ),
            criterion=e7.create_loss(TASK).to(device),
            optimizer=e7.create_optimizer(model, config.learning_rate),
            device=device,
        )
    finally:
        hook.remove()
    changed = any(
        not torch.equal(left, right.detach())
        for left, right in zip(before, model.parameters(), strict=True)
    )
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    result = {
        "status": "PASS",
        "task": TASK,
        "batch_size": config.batch_size,
        "training_config_valid": True,
        "authoritative_iterator": "src.training.harness.iter_production_batches",
        "training_helper": "src.training.harness.train_one_epoch",
        "logits_shape": list(shapes[0]) if len(shapes) == 1 else [],
        "train_batches": stats.batches,
        "train_examples": stats.examples,
        "loss_finite": math.isfinite(stats.loss),
        "gradients_finite": gradients_finite,
        "optimizer_step_succeeded": changed,
        "validation_accessed": False,
        "test_accessed": False,
    }
    if not all(
        (
            result["training_config_valid"],
            result["logits_shape"] == [batch_size, 15],
            result["train_batches"] == 1,
            result["train_examples"] == batch_size,
            result["loss_finite"],
            result["gradients_finite"],
            result["optimizer_step_succeeded"],
            result["validation_accessed"] is False,
            result["test_accessed"] is False,
        )
    ):
        raise BatchSizePilotError("selected batch failed production compatibility")
    return result


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def validate_observations(
    report: Mapping[str, Any],
    *,
    project_root: Path | None = None,
    verify_live_device: bool = False,
) -> None:
    required = {
        "stage",
        "pilot_version",
        "execution_id",
        "executed_at_utc",
        "executor",
        "elapsed_seconds",
        "contract_commit",
        "contract_source_sha256",
        "contract",
        "e9_source_sha256",
        "frozen_upstream_source_sha256",
        "device",
        "candidate_results",
        "selection",
        "production_compatibility",
        "finalized",
        "not_finalized",
        "validation_accessed",
        "test_accessed",
        "pilot_weights_preserved",
        "overall_status",
    }
    if set(report) != required:
        raise BatchSizePilotError("observations fields do not match the exact schema")
    if report.get("stage") != "E.9" or report.get("pilot_version") != PILOT_VERSION:
        raise BatchSizePilotError("observations have the wrong Stage E.9 identity")
    execution_id = report.get("execution_id")
    if not isinstance(execution_id, str) or not _EXECUTION_ID_PATTERN.fullmatch(
        execution_id
    ):
        raise BatchSizePilotError("observations have an invalid execution identity")
    if not isinstance(report.get("executed_at_utc"), str) or not report[
        "executed_at_utc"
    ].endswith("+00:00"):
        raise BatchSizePilotError("observations have an invalid UTC timestamp")
    if not isinstance(report.get("executor"), str) or not report["executor"].strip():
        raise BatchSizePilotError("observations lack an executor identity")
    if (
        isinstance(report.get("elapsed_seconds"), bool)
        or not isinstance(report.get("elapsed_seconds"), (int, float))
        or not math.isfinite(float(report["elapsed_seconds"]))
        or float(report["elapsed_seconds"]) <= 0
    ):
        raise BatchSizePilotError("observations have invalid elapsed time")
    if report.get("contract") != pilot_contract():
        raise BatchSizePilotError("observations do not match the frozen pilot contract")
    if report.get("contract_commit") != CONTRACT_COMMIT:
        raise BatchSizePilotError("observations have the wrong frozen contract commit")
    if report.get("contract_source_sha256") != {
        "PROJECT_HANDOFF.md@contract_commit": CONTRACT_HANDOFF_SHA256
    }:
        raise BatchSizePilotError("observations have the wrong contract source binding")

    source_hashes = report.get("e9_source_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or tuple(source_hashes) != OBSERVATION_SOURCE_PATHS
        or not all(_valid_sha256(value) for value in source_hashes.values())
    ):
        raise BatchSizePilotError("observations lack E.9 implementation provenance")
    upstream_hashes = report.get("frozen_upstream_source_sha256")
    if not isinstance(upstream_hashes, Mapping) or not upstream_hashes or not all(
        isinstance(key, str) and _valid_sha256(value)
        for key, value in upstream_hashes.items()
    ):
        raise BatchSizePilotError("observations lack frozen upstream provenance")
    if project_root is not None:
        root = Path(project_root).resolve()
        if dict(source_hashes) != observation_source_sha256(root):
            raise BatchSizePilotError("stored E.9 source hash does not match current source")
        if dict(upstream_hashes) != require_frozen_upstream(root):
            raise BatchSizePilotError("stored upstream hash does not match current authority")

    device = report.get("device")
    if not isinstance(device, Mapping) or set(device) != {
        "type",
        "name",
        "total_vram_bytes",
        "torch",
        "cuda_runtime",
    }:
        raise BatchSizePilotError("observations have invalid device identity")
    if (
        device.get("type") != "cuda"
        or device.get("name") != EXPECTED_DEVICE_NAME
        or device.get("total_vram_bytes") != EXPECTED_TOTAL_VRAM_BYTES
        or not isinstance(device.get("torch"), str)
        or not device["torch"]
        or not isinstance(device.get("cuda_runtime"), str)
        or not device["cuda_runtime"]
    ):
        raise BatchSizePilotError("observations do not bind the audited CUDA device")
    if verify_live_device:
        if not torch.cuda.is_available():
            raise BatchSizePilotError("live CUDA device verification was requested")
        properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if (
            str(properties.name) != device["name"]
            or int(properties.total_memory) != device["total_vram_bytes"]
        ):
            raise BatchSizePilotError("stored and live CUDA device identities differ")

    if report.get("validation_accessed") is not False:
        raise BatchSizePilotError("resource observations must not access validation")
    if report.get("test_accessed") is not False:
        raise BatchSizePilotError("resource observations must not access test")
    if report.get("pilot_weights_preserved") is not False:
        raise BatchSizePilotError("pilot-trained weights must be discarded")
    records = report.get("candidate_results")
    if not isinstance(records, list) or len(records) != len(BATCH_SIZE_CANDIDATES):
        raise BatchSizePilotError("candidate coverage is incomplete")
    if [record.get("batch_size") for record in records] != list(
        BATCH_SIZE_CANDIDATES
    ):
        raise BatchSizePilotError("candidate records are missing, duplicated, or reordered")
    config_identity = _config_identity_sha256()
    prefix_hashes: set[object] = set()
    for expected_batch, record in zip(BATCH_SIZE_CANDIDATES, records, strict=True):
        if not isinstance(record, Mapping):
            raise BatchSizePilotError("candidate record must be a mapping")
        if (
            record.get("execution_id") != execution_id
            or record.get("config_identity_sha256") != config_identity
        ):
            raise BatchSizePilotError("candidate execution/config identity is invalid")
        status = record.get("status")
        if status != "COMPLETE":
            if status not in {"CUDA_OOM", "SKIPPED_AFTER_SMALLER_OOM"}:
                raise BatchSizePilotError("candidate status is invalid")
            if (
                record.get("resource_eligible") is not False
                or record.get("warmup_steps_completed") != 0
                or record.get("measured_steps_completed") != 0
                or record.get("losses_finite") is not False
                or record.get("gradients_finite") is not False
                or record.get("optimizer_steps_succeeded") is not False
                or record.get("total_vram_bytes") != device["total_vram_bytes"]
            ):
                raise BatchSizePilotError("failed candidate accounting is invalid")
            continue
        prefix_hashes.add(record.get("first_32_source_indices_sha256"))
        if record.get("warmup_steps_completed") != WARMUP_STEPS:
            raise BatchSizePilotError("candidate warmup count is invalid")
        if record.get("measured_steps_completed") != MEASURED_STEPS:
            raise BatchSizePilotError("candidate measurement count is invalid")
        timing_raw = record.get("latency_ms_samples")
        throughput_raw = record.get("examples_per_second_samples")
        losses_raw = record.get("measured_loss_samples")
        if not all(isinstance(values, list) for values in (timing_raw, throughput_raw, losses_raw)):
            raise BatchSizePilotError("candidate raw measurement arrays are missing")
        if not all(len(values) == MEASURED_STEPS for values in (timing_raw, throughput_raw, losses_raw)):
            raise BatchSizePilotError("candidate raw measurement counts are invalid")
        try:
            timing = [float(value) for value in timing_raw]
            throughput = [float(value) for value in throughput_raw]
            losses = [float(value) for value in losses_raw]
        except (TypeError, ValueError) as exc:
            raise BatchSizePilotError("candidate measurements are not numeric") from exc
        if not all(math.isfinite(value) and value > 0 for value in (*timing, *throughput)):
            raise BatchSizePilotError("candidate timing or throughput is invalid")
        if not all(math.isfinite(value) for value in losses):
            raise BatchSizePilotError("candidate measured loss is nonfinite")
        derived_throughput = [expected_batch * 1000.0 / value for value in timing]
        if any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(throughput, derived_throughput, strict=True)
        ):
            raise BatchSizePilotError("candidate throughput samples are inconsistent")
        expected_summaries = (
            (record.get("latency_ms_median"), statistics.median(timing)),
            (record.get("latency_ms_p95_nearest_rank"), _nearest_rank_p95(timing)),
            (
                record.get("median_examples_per_second"),
                statistics.median(derived_throughput),
            ),
        )
        if any(
            isinstance(stored, bool)
            or not isinstance(stored, (int, float))
            or not math.isclose(
                float(stored), expected, rel_tol=1e-12, abs_tol=1e-12
            )
            for stored, expected in expected_summaries
        ):
            raise BatchSizePilotError("candidate timing/throughput summary is invalid")
        memory_fields = (
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "peak_allocated_delta_bytes",
            "peak_reserved_delta_bytes",
            "total_vram_bytes",
        )
        if any(
            isinstance(record.get(field), bool)
            or not isinstance(record.get(field), int)
            or int(record[field]) < 0
            for field in memory_fields
        ):
            raise BatchSizePilotError("candidate CUDA memory accounting is invalid")
        if (
            record["total_vram_bytes"] != device["total_vram_bytes"]
            or record["total_vram_bytes"] <= 0
            or record["baseline_reserved_bytes"] < record["baseline_allocated_bytes"]
            or record["peak_allocated_bytes"] < record["baseline_allocated_bytes"]
            or record["peak_reserved_bytes"] < record["baseline_reserved_bytes"]
            or record["peak_reserved_bytes"] < record["peak_allocated_bytes"]
            or record["peak_allocated_delta_bytes"]
            != record["peak_allocated_bytes"] - record["baseline_allocated_bytes"]
            or record["peak_reserved_delta_bytes"]
            != record["peak_reserved_bytes"] - record["baseline_reserved_bytes"]
        ):
            raise BatchSizePilotError("candidate CUDA memory relationships are invalid")
        peak_fraction = record["peak_reserved_bytes"] / record["total_vram_bytes"]
        if (
            not isinstance(record.get("peak_reserved_vram_fraction"), (int, float))
            or not math.isclose(
                float(record["peak_reserved_vram_fraction"]),
                peak_fraction,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not isinstance(record.get("peak_reserved_vram_percent"), (int, float))
            or not math.isclose(
                float(record["peak_reserved_vram_percent"]),
                100.0 * peak_fraction,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or record.get("headroom_threshold_fraction") != HEADROOM_THRESHOLD
        ):
            raise BatchSizePilotError("candidate peak reserved percentage is invalid")
        if (
            record.get("parameter_count") != e7.EXPECTED_PARAMETERS[TASK]
            or record.get("timing_scope")
            != "host_to_device_forward_loss_backward_optimizer_step"
            or record.get("cpu_d4_d6_e2_construction_timed") is not False
            or record.get("warmups_excluded_from_timing") is not True
            or not _valid_sha256(record.get("first_32_source_indices_sha256"))
            or not _valid_sha256(record.get("all_candidate_source_indices_sha256"))
        ):
            raise BatchSizePilotError("candidate execution provenance is invalid")
        derived_eligible = bool(
            record.get("losses_finite") is True
            and record.get("gradients_finite") is True
            and record.get("optimizer_steps_succeeded") is True
            and peak_fraction < HEADROOM_THRESHOLD
        )
        if record.get("resource_eligible") is not derived_eligible:
            raise BatchSizePilotError("candidate resource eligibility is invalid")
    if len(prefix_hashes) != 1:
        raise BatchSizePilotError("candidates did not restart the same seeded traversal")

    selection = select_batch_size(records)
    if report.get("selection") != selection:
        raise BatchSizePilotError("stored batch-size selection is not reproducible")
    selected = selection["selected_batch_size"]
    if report.get("finalized") != {"htgnn_training_batch_size": selected}:
        raise BatchSizePilotError("finalized batch size is inconsistent")
    compatibility = report.get("production_compatibility")
    if not isinstance(compatibility, Mapping) or not all(
        (
            compatibility.get("status") == "PASS",
            compatibility.get("task") == TASK,
            compatibility.get("batch_size") == selected,
            compatibility.get("training_config_valid") is True,
            compatibility.get("authoritative_iterator")
            == "src.training.harness.iter_production_batches",
            compatibility.get("training_helper")
            == "src.training.harness.train_one_epoch",
            compatibility.get("logits_shape") == [selected, 15],
            compatibility.get("train_batches") == 1,
            compatibility.get("train_examples") == selected,
            compatibility.get("loss_finite") is True,
            compatibility.get("gradients_finite") is True,
            compatibility.get("optimizer_step_succeeded") is True,
            compatibility.get("validation_accessed") is False,
            compatibility.get("test_accessed") is False,
        )
    ):
        raise BatchSizePilotError("selected batch lacks production compatibility")
    if report.get("not_finalized") != [
        "learning_rate",
        "weighted_loss_configuration",
        "task_performance",
        "final_model_checkpoint",
        "test_performance",
    ]:
        raise BatchSizePilotError("Stage E.9 finalization scope is invalid")
    if report.get("overall_status") != "PASS":
        raise BatchSizePilotError("Stage E.9 observations are not passing")


def validate_observation_file(
    path: Path,
    *,
    project_root: Path,
    expected_sha256: str,
    verify_live_device: bool = False,
) -> Mapping[str, Any]:
    """Validate exact observation bytes, then the production evidence schema."""

    observation_path = Path(path)
    if not _valid_sha256(expected_sha256) or _sha256(observation_path) != expected_sha256:
        raise BatchSizePilotError("observation file SHA-256 binding is invalid")
    try:
        report = json.loads(observation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BatchSizePilotError(f"cannot read observation file: {exc}") from exc
    if not isinstance(report, Mapping):
        raise BatchSizePilotError("observation file root must be a mapping")
    validate_observations(
        report,
        project_root=project_root,
        verify_live_device=verify_live_device,
    )
    return report


def run_pilot(project_root: Path, *, executor: str = "Codex") -> dict[str, Any]:
    """Execute all four bounded candidates and finalize only batch size."""

    if not isinstance(executor, str) or not executor.strip():
        raise ValueError("executor must be a non-empty string")
    started_at = datetime.now(timezone.utc)
    execution_id = started_at.strftime("e9-%Y%m%dt%H%M%Sz")
    started_clock = time.perf_counter()
    root = Path(project_root).resolve()
    frozen_hashes = require_frozen_upstream(root)
    if not torch.cuda.is_available():
        raise BatchSizePilotError("CUDA is required for the Stage E.9 resource pilot")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    total_vram_bytes = int(properties.total_memory)
    dataset = d4.LazyGraphDataset(
        TASK,
        "train",
        processed_root=root / "data" / "processed" / "preprocessed",
    )
    records: list[dict[str, Any]] = []
    smaller_oom = False
    for batch_size in BATCH_SIZE_CANDIDATES:
        if smaller_oom:
            records.append(
                _candidate_failure_record(
                    batch_size,
                    execution_id=execution_id,
                    status="SKIPPED_AFTER_SMALLER_OOM",
                    failure_type="SafetyStop",
                    failure_message=(
                        "not attempted after a smaller approved candidate exhausted CUDA memory"
                    ),
                    total_vram_bytes=total_vram_bytes,
                )
            )
            continue
        record = run_candidate(
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            total_vram_bytes=total_vram_bytes,
            execution_id=execution_id,
        )
        records.append(record)
        smaller_oom = record["status"] == "CUDA_OOM"
    selection = select_batch_size(records)
    production_compatibility = verify_production_compatibility(
        dataset,
        batch_size=selection["selected_batch_size"],
        device=device,
    )
    report = {
        "stage": "E.9",
        "pilot_version": PILOT_VERSION,
        "execution_id": execution_id,
        "executed_at_utc": started_at.isoformat(),
        "executor": executor.strip(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "contract_commit": CONTRACT_COMMIT,
        "contract_source_sha256": {
            "PROJECT_HANDOFF.md@contract_commit": CONTRACT_HANDOFF_SHA256
        },
        "contract": pilot_contract(),
        "e9_source_sha256": observation_source_sha256(root),
        "frozen_upstream_source_sha256": frozen_hashes,
        "device": {
            "type": "cuda",
            "name": str(properties.name),
            "total_vram_bytes": total_vram_bytes,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "candidate_results": records,
        "selection": selection,
        "production_compatibility": production_compatibility,
        "finalized": {"htgnn_training_batch_size": selection["selected_batch_size"]},
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
    validate_observations(report, project_root=root, verify_live_device=True)
    return report


def write_observations(
    report: Mapping[str, Any], path: Path, *, project_root: Path
) -> str:
    """Write noisy runtime observations without canonicalizing their values."""

    validate_observations(report, project_root=project_root, verify_live_device=True)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(report), ensure_ascii=False, indent=2))
        handle.write("\n")
    return _sha256(output)
