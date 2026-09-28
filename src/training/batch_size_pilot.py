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
import statistics
import time
from typing import Any, Final, Iterator, Mapping, Sequence

import torch

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import pyg_adapter as e2
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


class BatchSizePilotError(RuntimeError):
    """Raised when Stage E.9 resource evidence violates its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    handoff = root / "PROJECT_HANDOFF.md"
    if not handoff.is_file():
        raise BatchSizePilotError("PROJECT_HANDOFF.md is missing")
    hashes["PROJECT_HANDOFF.md"] = _sha256(handoff)
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
    status: str,
    failure_type: str,
    failure_message: str,
    total_vram_bytes: int,
) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
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
) -> dict[str, Any]:
    """Run one isolated candidate and return observational measurements."""

    if device.type != "cuda":
        raise BatchSizePilotError("Stage E.9 batch-size measurement requires CUDA")
    if dataset.experiment != TASK or dataset.split != "train":
        raise BatchSizePilotError("pilot dataset must be authoritative E2/train")
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
            "headroom_threshold_fraction": HEADROOM_THRESHOLD,
        }
    except torch.OutOfMemoryError as exc:
        return _candidate_failure_record(
            batch_size,
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


def validate_observations(report: Mapping[str, Any]) -> None:
    required = {
        "stage",
        "pilot_version",
        "execution_id",
        "executed_at_utc",
        "executor",
        "elapsed_seconds",
        "contract_commit",
        "contract",
        "e9_source_sha256",
        "device",
        "candidate_results",
        "selection",
        "overall_status",
    }
    if not required.issubset(report):
        raise BatchSizePilotError("observations are missing required fields")
    if report.get("stage") != "E.9" or report.get("pilot_version") != PILOT_VERSION:
        raise BatchSizePilotError("observations have the wrong Stage E.9 identity")
    if not all(
        isinstance(report.get(field), str) and bool(report[field])
        for field in ("execution_id", "executed_at_utc", "executor")
    ):
        raise BatchSizePilotError("observations lack execution identity")
    if not isinstance(report.get("elapsed_seconds"), (int, float)) or not (
        math.isfinite(float(report["elapsed_seconds"]))
        and float(report["elapsed_seconds"]) > 0
    ):
        raise BatchSizePilotError("observations have invalid elapsed time")
    if report.get("contract") != pilot_contract():
        raise BatchSizePilotError("observations do not match the frozen pilot contract")
    if report.get("contract_commit") != CONTRACT_COMMIT:
        raise BatchSizePilotError("observations have the wrong frozen contract commit")
    source_hashes = report.get("e9_source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != {
        "src/training/batch_size_pilot.py",
        "scripts/run_stage_e9_batch_size_pilot.py",
    }:
        raise BatchSizePilotError("observations lack E.9 implementation provenance")
    if report.get("validation_accessed") is not False:
        raise BatchSizePilotError("resource observations must not access validation")
    if report.get("test_accessed") is not False:
        raise BatchSizePilotError("resource observations must not access test")
    if report.get("pilot_weights_preserved") is not False:
        raise BatchSizePilotError("pilot-trained weights must be discarded")
    records = report.get("candidate_results")
    if not isinstance(records, list):
        raise BatchSizePilotError("candidate_results must be a list")
    selection = select_batch_size(records)
    if report.get("selection") != selection:
        raise BatchSizePilotError("stored batch-size selection is not reproducible")
    prefix_hashes = {
        record.get("first_32_source_indices_sha256")
        for record in records
        if record.get("status") == "COMPLETE"
    }
    if len(prefix_hashes) != 1:
        raise BatchSizePilotError("candidates did not restart the same seeded traversal")
    for record in records:
        if record.get("status") == "COMPLETE":
            if record.get("warmup_steps_completed") != WARMUP_STEPS:
                raise BatchSizePilotError("candidate warmup count is invalid")
            if record.get("measured_steps_completed") != MEASURED_STEPS:
                raise BatchSizePilotError("candidate measurement count is invalid")
            if len(record.get("latency_ms_samples", [])) != MEASURED_STEPS:
                raise BatchSizePilotError("candidate timing samples are incomplete")
            timing = [float(value) for value in record["latency_ms_samples"]]
            throughput = [
                float(value)
                for value in record.get("examples_per_second_samples", [])
            ]
            if len(throughput) != MEASURED_STEPS or not all(
                math.isfinite(value) and value > 0
                for value in (*timing, *throughput)
            ):
                raise BatchSizePilotError("candidate timing or throughput is invalid")
            if not math.isclose(
                float(record.get("latency_ms_median", math.nan)),
                statistics.median(timing),
            ) or not math.isclose(
                float(record.get("latency_ms_p95_nearest_rank", math.nan)),
                _nearest_rank_p95(timing),
            ):
                raise BatchSizePilotError("candidate latency summary is invalid")
            if not math.isclose(
                float(record.get("median_examples_per_second", math.nan)),
                statistics.median(throughput),
            ):
                raise BatchSizePilotError("candidate throughput summary is invalid")
            memory_fields = (
                "baseline_allocated_bytes",
                "baseline_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "total_vram_bytes",
            )
            if any(
                isinstance(record.get(field), bool)
                or not isinstance(record.get(field), int)
                or int(record[field]) < 0
                for field in memory_fields
            ):
                raise BatchSizePilotError("candidate CUDA memory accounting is invalid")
            peak_fraction = float(record.get("peak_reserved_vram_fraction", math.nan))
            if not math.isclose(
                peak_fraction,
                int(record["peak_reserved_bytes"]) / int(record["total_vram_bytes"]),
            ):
                raise BatchSizePilotError("candidate peak reserved fraction is invalid")
            derived_eligible = bool(
                record.get("losses_finite") is True
                and record.get("gradients_finite") is True
                and record.get("optimizer_steps_succeeded") is True
                and peak_fraction < HEADROOM_THRESHOLD
            )
            if record.get("resource_eligible") is not derived_eligible:
                raise BatchSizePilotError("candidate resource eligibility is invalid")
            if record.get("resource_eligible") is True and not (
                peak_fraction < HEADROOM_THRESHOLD
            ):
                raise BatchSizePilotError("eligible candidate violates VRAM headroom")
    if report.get("overall_status") != "PASS":
        raise BatchSizePilotError("Stage E.9 observations are not passing")


def run_pilot(project_root: Path, *, executor: str = "Codex") -> dict[str, Any]:
    """Execute all four bounded candidates and finalize only batch size."""

    if not isinstance(executor, str) or not executor.strip():
        raise ValueError("executor must be a non-empty string")
    started_at = datetime.now(timezone.utc)
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
        )
        records.append(record)
        smaller_oom = record["status"] == "CUDA_OOM"
    selection = select_batch_size(records)
    report = {
        "stage": "E.9",
        "pilot_version": PILOT_VERSION,
        "execution_id": started_at.strftime("e9-%Y%m%dt%H%M%Sz"),
        "executed_at_utc": started_at.isoformat(),
        "executor": executor.strip(),
        "elapsed_seconds": time.perf_counter() - started_clock,
        "contract_commit": CONTRACT_COMMIT,
        "contract": pilot_contract(),
        "e9_source_sha256": {
            "src/training/batch_size_pilot.py": _sha256(
                root / "src" / "training" / "batch_size_pilot.py"
            ),
            "scripts/run_stage_e9_batch_size_pilot.py": _sha256(
                root / "scripts" / "run_stage_e9_batch_size_pilot.py"
            ),
        },
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
    validate_observations(report)
    return report


def write_observations(report: Mapping[str, Any], path: Path) -> None:
    """Write noisy runtime observations without canonicalizing their values."""

    validate_observations(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(report), ensure_ascii=False, indent=2))
        handle.write("\n")
