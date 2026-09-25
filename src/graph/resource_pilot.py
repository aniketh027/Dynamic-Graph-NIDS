"""Stage E.6 bounded real-data forward and resource pilot.

This module measures the frozen D4 -> D6 -> E.2 -> E.5 inference path.  It
does not train, optimize, score predictions, choose a batch size, or mutate the
frozen E.2-E.5 architecture.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gc
import hashlib
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable, Final, Mapping, Protocol, TypeVar

import numpy as np
import psutil
import torch
import torch_geometric
from torch_geometric.data import HeteroData

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import htgnn_classifier as e5
from src.graph import pyg_adapter as e2


PILOT_VERSION: Final = "stage-e6-bounded-resource-pilot-v1"
TASKS: Final = ("E1", "E2", "E3")
MEASUREMENT_MODES: Final = (
    "e2_conversion",
    "host_to_device",
    "normal_forward",
    "attention_forward",
)
DEFAULT_BATCH_SIZE: Final = 32
DEFAULT_WARMUP_ITERATIONS: Final = 5
DEFAULT_MEASURED_ITERATIONS: Final = 20
MAX_BATCH_SIZE: Final = 64
MAX_WARMUP_ITERATIONS: Final = 10
MAX_MEASURED_ITERATIONS: Final = 50
SEED: Final = 20260924
CUDA_EQUIVALENCE_RTOL: Final = 1e-5
CUDA_EQUIVALENCE_ATOL: Final = 1e-6
EXPECTED_PARAMETERS: Final = {"E1": 7183, "E2": 7183, "E3": 6721}
EXPECTED_LOGIT_WIDTHS: Final = {"E1": 15, "E2": 15, "E3": 1}
FROZEN_SOURCE_SHA256: Final = {
    "src/graph/pyg_adapter.py": (
        "32097a6af3640522595b99e03a2d82906cfaae7d930eb93e4e13d451b5d9fb20"
    ),
    "src/graph/feature_encoder.py": (
        "83f7352e3ee4ff8e3e3b607f28cdadb02ec91a98ff552a29fb226de0d881fdca"
    ),
    "src/graph/hierarchical_encoder.py": (
        "23107e5bd362107754ff5d5883e74b334f54adc3c07891aef724665086d92985"
    ),
    "src/graph/htgnn_classifier.py": (
        "dc72ca252904cd3dcb8b894b6e06e02696cd60529d808c18509520cf25432248"
    ),
}


class ResourcePilotContractError(RuntimeError):
    """Raised when the bounded E.6 measurement contract is not satisfied."""


@dataclass(frozen=True)
class PilotConfig:
    """Strict bounds for one E.6 pilot run."""

    batch_size: int = DEFAULT_BATCH_SIZE
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    measured_iterations: int = DEFAULT_MEASURED_ITERATIONS

    def __post_init__(self) -> None:
        _require_bounded_int("batch_size", self.batch_size, 1, MAX_BATCH_SIZE)
        _require_bounded_int(
            "warmup_iterations",
            self.warmup_iterations,
            1,
            MAX_WARMUP_ITERATIONS,
        )
        _require_bounded_int(
            "measured_iterations",
            self.measured_iterations,
            2,
            MAX_MEASURED_ITERATIONS,
        )


def _require_bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def bounded_sample_indices(dataset_length: int, batch_size: int) -> tuple[int, ...]:
    """Choose a fixed-size, capture-order-spanning set without split iteration."""

    _require_bounded_int("batch_size", batch_size, 1, MAX_BATCH_SIZE)
    if isinstance(dataset_length, bool) or not isinstance(dataset_length, int):
        raise TypeError("dataset_length must be an integer")
    if dataset_length < batch_size:
        raise ResourcePilotContractError(
            "dataset is smaller than the requested bounded pilot batch"
        )
    if batch_size == 1:
        return (dataset_length // 2,)
    indices = tuple(
        (position * (dataset_length - 1)) // (batch_size - 1)
        for position in range(batch_size)
    )
    if len(set(indices)) != batch_size:
        raise ResourcePilotContractError("bounded sample selection produced duplicates")
    return indices


def build_bounded_real_train_batch(
    task: str,
    config: PilotConfig,
    project_root: Path,
) -> tuple[d4.LazyGraphDataset, d6.LogicalGraphBatch, tuple[int, ...]]:
    """Use only the authoritative lazy D4/D6 train path for one bounded batch."""

    if task not in TASKS:
        raise ValueError("task must be exactly one of E1, E2, or E3")
    dataset = d4.LazyGraphDataset(
        task,
        "train",
        processed_root=Path(project_root) / "data" / "processed" / "preprocessed",
    )
    if not dataset.is_lazy_read_only:
        raise ResourcePilotContractError("D4 source is not lazy and read-only")
    indices = bounded_sample_indices(len(dataset), config.batch_size)
    batch = d6.build_logical_batch(dataset, indices)
    if batch.batch_size != config.batch_size:
        raise ResourcePilotContractError("D6 returned an unexpected batch size")
    return dataset, batch, indices


def create_frozen_model(task: str) -> e5.HTGNNClassifier:
    """Construct the unchanged task-specific frozen E.5 model in eval mode."""

    if task not in TASKS:
        raise ValueError("task must be exactly one of E1, E2, or E3")
    model = e5.HTGNNClassifier(task).eval()
    measured = sum(parameter.numel() for parameter in model.parameters())
    if measured != EXPECTED_PARAMETERS[task]:
        raise ResourcePilotContractError(
            f"{task} parameter count changed: {measured} != {EXPECTED_PARAMETERS[task]}"
        )
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_source_hashes(project_root: Path) -> dict[str, str]:
    return {
        relative: _sha256(Path(project_root) / relative)
        for relative in FROZEN_SOURCE_SHA256
    }


def require_frozen_sources(project_root: Path) -> dict[str, str]:
    measured = frozen_source_hashes(project_root)
    mismatches = {
        path: {"expected": FROZEN_SOURCE_SHA256[path], "measured": digest}
        for path, digest in measured.items()
        if digest != FROZEN_SOURCE_SHA256[path]
    }
    if mismatches:
        raise ResourcePilotContractError(
            f"frozen E.2-E.5 source changed: {mismatches}"
        )
    return measured


def _data_snapshot(data: HeteroData) -> dict[str, Any]:
    return {
        "feature_x": data["feature"].x.detach().cpu().clone(),
        "feature_id": data["feature"].feature_id.detach().cpu().clone(),
        "feature_batch": data["feature"].batch.detach().cpu().clone(),
        "category_id": data["category"].category_id.detach().cpu().clone(),
        "category_batch": data["category"].batch.detach().cpu().clone(),
        "sample_y": data["sample"].y.detach().cpu().clone(),
        "sample_batch": data["sample"].batch.detach().cpu().clone(),
        "feature_edges": data[e2.FEATURE_RELATION].edge_index.detach().cpu().clone(),
        "category_edges": data[e2.CATEGORY_RELATION].edge_index.detach().cpu().clone(),
        "audit": deepcopy(data.e2_audit),
    }


def _data_matches_snapshot(data: HeteroData, snapshot: Mapping[str, Any]) -> bool:
    current = _data_snapshot(data)
    return bool(
        all(
            torch.equal(current[key], snapshot[key])
            for key in current
            if key != "audit"
        )
        and current["audit"] == snapshot["audit"]
    )


def verify_forward_contract(
    task: str,
    model: e5.HTGNNClassifier,
    data: HeteroData,
) -> dict[str, Any]:
    """Check E.6 correctness without labels, prediction metrics, or gradients."""

    if model.task != task:
        raise ResourcePilotContractError("task-model selection mismatch")
    snapshot = _data_snapshot(data)
    parameter_snapshot = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    with torch.inference_mode():
        normal_logits = model(data)
        attention_logits, attention = model(data, return_attention=True)
    bitwise_identical = bool(torch.equal(normal_logits, attention_logits))
    if normal_logits.device.type == "cuda":
        attention_equivalent = bool(
            torch.allclose(
                normal_logits,
                attention_logits,
                rtol=CUDA_EQUIVALENCE_RTOL,
                atol=CUDA_EQUIVALENCE_ATOL,
            )
        )
        comparison_policy = (
            "CUDA torch.allclose with frozen E.5 rtol=1e-5 and atol=1e-6"
        )
    else:
        attention_equivalent = bitwise_identical
        comparison_policy = "CPU bitwise equality"
    batch_size = int(data["sample"].num_nodes)
    expected_shape = (
        (batch_size,)
        if task == "E3"
        else (batch_size, EXPECTED_LOGIT_WIDTHS[task])
    )
    feature_edges, feature_alpha = attention[e2.FEATURE_RELATION]
    category_edges, category_alpha = attention[e2.CATEGORY_RELATION]
    source_unchanged = _data_matches_snapshot(data, snapshot)
    parameters_unchanged = all(
        torch.equal(value.detach().cpu(), parameter_snapshot[key])
        for key, value in model.state_dict().items()
    )
    facts = {
        "task_model_match": model.task == task,
        "logit_shape": list(normal_logits.shape),
        "expected_logit_shape": list(expected_shape),
        "finite_logits": bool(torch.isfinite(normal_logits).all()),
        "attention_logits_identical": attention_equivalent,
        "attention_logit_comparison_policy": comparison_policy,
        "attention_logits_bitwise_identical_observation": bitwise_identical,
        "feature_attention_shape": list(feature_alpha.shape),
        "expected_feature_attention_shape": [batch_size * 78, 4],
        "category_attention_shape": list(category_alpha.shape),
        "expected_category_attention_shape": [batch_size * 8, 4],
        "feature_topology_identical": bool(
            torch.equal(
                feature_edges,
                data[e2.FEATURE_RELATION].edge_index,
            )
        ),
        "category_topology_identical": bool(
            torch.equal(
                category_edges,
                data[e2.CATEGORY_RELATION].edge_index,
            )
        ),
        "source_tensors_unchanged": source_unchanged,
        "model_state_unchanged": parameters_unchanged,
    }
    facts["status"] = (
        "PASS"
        if all(
            (
                facts["task_model_match"],
                facts["logit_shape"] == facts["expected_logit_shape"],
                facts["finite_logits"],
                facts["attention_logits_identical"],
                facts["feature_attention_shape"]
                == facts["expected_feature_attention_shape"],
                facts["category_attention_shape"]
                == facts["expected_category_attention_shape"],
                facts["feature_topology_identical"],
                facts["category_topology_identical"],
                facts["source_tensors_unchanged"],
                facts["model_state_unchanged"],
            )
        )
        else "FAIL"
    )
    if facts["status"] != "PASS":
        raise ResourcePilotContractError(f"{task} forward contract failed: {facts}")
    return facts


def summarize_timings_ms(samples_ms: list[float]) -> dict[str, Any]:
    if len(samples_ms) < 2:
        raise ValueError("at least two timing samples are required")
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    rounded = [round(value, 6) for value in samples_ms]
    return {
        "unit": "ms",
        "samples": rounded,
        "minimum": round(ordered[0], 6),
        "median": round(statistics.median(ordered), 6),
        "p95_nearest_rank": round(ordered[p95_index], 6),
        "maximum": round(ordered[-1], 6),
    }


T = TypeVar("T")


def measure_cpu_callable(
    operation: Callable[[], T],
    measured_iterations: int,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[T, dict[str, Any]]:
    """Measure one first call and repeated CPU calls with process RSS context."""

    _require_bounded_int(
        "measured_iterations", measured_iterations, 2, MAX_MEASURED_ITERATIONS
    )
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = clock_ns()
    result = operation()
    first_ms = (clock_ns() - start) / 1_000_000.0
    rss_after_first = process.memory_info().rss
    repeated_ms: list[float] = []
    peak_rss = max(rss_before, rss_after_first)
    for _ in range(measured_iterations):
        start = clock_ns()
        result = operation()
        repeated_ms.append((clock_ns() - start) / 1_000_000.0)
        peak_rss = max(peak_rss, process.memory_info().rss)
    return result, {
        "first_call_ms": round(first_ms, 6),
        "repeated_calls": summarize_timings_ms(repeated_ms),
        "rss_before_bytes": int(rss_before),
        "rss_after_first_bytes": int(rss_after_first),
        "peak_observed_rss_bytes": int(peak_rss),
        "first_call_rss_delta_bytes": int(rss_after_first - rss_before),
        "peak_observed_rss_delta_bytes": int(peak_rss - rss_before),
        "rss_note": "process RSS is allocator- and OS-sensitive observational context",
    }


class CudaRuntime(Protocol):
    def synchronize(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def memory_allocated(self) -> int: ...

    def memory_reserved(self) -> int: ...

    def max_memory_allocated(self) -> int: ...

    def max_memory_reserved(self) -> int: ...


class TorchCudaRuntime:
    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def memory_allocated(self) -> int:
        return int(torch.cuda.memory_allocated())

    def memory_reserved(self) -> int:
        return int(torch.cuda.memory_reserved())

    def max_memory_allocated(self) -> int:
        return int(torch.cuda.max_memory_allocated())

    def max_memory_reserved(self) -> int:
        return int(torch.cuda.max_memory_reserved())


def measure_cuda_callable(
    operation: Callable[[Any], T],
    *,
    warmup_iterations: int,
    measured_iterations: int,
    prepare: Callable[[], Any] | None = None,
    cuda_runtime: CudaRuntime | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[T, dict[str, Any]]:
    """Measure completed CUDA work with explicit synchronization and peak reset.

    ``prepare`` runs outside the timed interval.  This keeps the CPU clone used
    for repeated H2D trials out of transfer timing while preserving the source.
    """

    _require_bounded_int(
        "warmup_iterations", warmup_iterations, 1, MAX_WARMUP_ITERATIONS
    )
    _require_bounded_int(
        "measured_iterations", measured_iterations, 2, MAX_MEASURED_ITERATIONS
    )
    runtime = TorchCudaRuntime() if cuda_runtime is None else cuda_runtime
    result: T | None = None
    for _ in range(warmup_iterations):
        prepared = prepare() if prepare is not None else None
        result = operation(prepared)
        runtime.synchronize()
    # Warmup outputs are not part of the steady-state allocation baseline.
    # Release them before resetting peaks so each reported mode measures its
    # own output/activation requirement rather than retaining the last warmup.
    result = None
    prepared = None
    gc.collect()
    runtime.synchronize()
    runtime.reset_peak_memory_stats()
    allocated_before = runtime.memory_allocated()
    reserved_before = runtime.memory_reserved()
    timings_ms: list[float] = []
    for _ in range(measured_iterations):
        # Do not overlap the prior call's returned tensors with the next call.
        # The mode's internal temporaries remain visible to CUDA peak counters.
        result = None
        prepared = prepare() if prepare is not None else None
        runtime.synchronize()
        start = clock_ns()
        result = operation(prepared)
        runtime.synchronize()
        timings_ms.append((clock_ns() - start) / 1_000_000.0)
    if result is None:  # pragma: no cover - protected by positive bounds
        raise ResourcePilotContractError("CUDA measurement produced no result")
    allocated_after = runtime.memory_allocated()
    reserved_after = runtime.memory_reserved()
    peak_allocated = runtime.max_memory_allocated()
    peak_reserved = runtime.max_memory_reserved()
    return result, {
        "timing": summarize_timings_ms(timings_ms),
        "timing_method": (
            "host monotonic clock with CUDA synchronization immediately before "
            "and after each measured operation"
        ),
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": reserved_after,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(0, peak_allocated - allocated_before),
        "peak_reserved_delta_bytes": max(0, peak_reserved - reserved_before),
    }


def environment_identity() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ResourcePilotContractError("CUDA is required for the real E.6 pilot")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "gpu_free_memory_at_pilot_start_bytes": int(free_bytes),
        "gpu_total_memory_at_pilot_start_bytes": int(total_bytes),
    }


def _throughput(batch_size: int, median_ms: float) -> float:
    if median_ms <= 0:
        raise ResourcePilotContractError("non-positive measured latency")
    return round(batch_size * 1000.0 / median_ms, 3)


def _measure_one_task(
    task: str,
    config: PilotConfig,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset, batch, indices = build_bounded_real_train_batch(
        task, config, project_root
    )
    batch_snapshot = np.array(batch.feature_x, copy=True)
    data, conversion = measure_cpu_callable(
        lambda: e2.logical_batch_to_heterodata(batch, project_root=project_root),
        config.measured_iterations,
    )
    _, provenance_validation = measure_cpu_callable(
        lambda: e2._load_frozen_identities(project_root),  # noqa: SLF001
        config.measured_iterations,
    )
    if not np.array_equal(batch.feature_x, batch_snapshot, equal_nan=True):
        raise ResourcePilotContractError("E.2 conversion mutated the D6 source")

    gc.collect()
    torch.cuda.empty_cache()
    cuda_data, transfer = measure_cuda_callable(
        lambda prepared: prepared.to("cuda", non_blocking=False),
        prepare=data.clone,
        warmup_iterations=config.warmup_iterations,
        measured_iterations=config.measured_iterations,
    )
    torch.manual_seed(SEED)
    model = create_frozen_model(task).to("cuda")
    contract = verify_forward_contract(task, model, cuda_data)
    with torch.inference_mode():
        normal_result, normal = measure_cuda_callable(
            lambda _prepared: model(cuda_data),
            warmup_iterations=config.warmup_iterations,
            measured_iterations=config.measured_iterations,
        )
        attention_result, attention = measure_cuda_callable(
            lambda _prepared: model(cuda_data, return_attention=True),
            warmup_iterations=config.warmup_iterations,
            measured_iterations=config.measured_iterations,
        )
    measured_bitwise_identical = bool(
        torch.equal(normal_result, attention_result[0])
    )
    measured_max_abs_difference = float(
        torch.max(torch.abs(normal_result - attention_result[0])).item()
    )
    if not torch.allclose(
        normal_result,
        attention_result[0],
        rtol=CUDA_EQUIVALENCE_RTOL,
        atol=CUDA_EQUIVALENCE_ATOL,
    ):
        raise ResourcePilotContractError(
            f"{task} measured normal and attention logits differ"
        )
    conversion_median = conversion["repeated_calls"]["median"]
    provenance_median = provenance_validation["repeated_calls"]["median"]
    provenance_ratio = (
        provenance_median / conversion_median if conversion_median > 0 else 0.0
    )
    conversion["throughput_samples_per_second"] = _throughput(
        config.batch_size, conversion_median
    )
    transfer["throughput_samples_per_second"] = _throughput(
        config.batch_size, transfer["timing"]["median"]
    )
    normal["throughput_samples_per_second"] = _throughput(
        config.batch_size, normal["timing"]["median"]
    )
    attention["throughput_samples_per_second"] = _throughput(
        config.batch_size, attention["timing"]["median"]
    )
    provenance_observation = {
        "measurement_scope": (
            "frozen E.2 provenance/report load and validation component; "
            "no caching or adapter modification"
        ),
        "first_call_ms": provenance_validation["first_call_ms"],
        "repeated_calls": provenance_validation["repeated_calls"],
        "median_fraction_of_conversion": round(provenance_ratio, 6),
        "appears_measurable": bool(
            provenance_median >= 0.05 and provenance_ratio >= 0.05
        ),
        "reporting_heuristic": (
            "component median >= 0.05 ms and >= 5% of conversion median"
        ),
    }
    deterministic = {
        "task": task,
        "split": "train",
        "source_dataset_length": len(dataset),
        "sample_indices": list(indices),
        "batch_size": batch.batch_size,
        "d4_source_lazy_read_only": dataset.is_lazy_read_only,
        "d6_batch_read_only": not batch.feature_x.flags.writeable,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "expected_parameter_count": EXPECTED_PARAMETERS[task],
        "forward_contract": {
            key: value
            for key, value in contract.items()
            if key != "attention_logits_bitwise_identical_observation"
        },
        "measurement_modes_completed": list(MEASUREMENT_MODES),
        "no_full_split_materialization": True,
    }
    observational = {
        "task": task,
        "e2_conversion": conversion,
        "e2_provenance_validation_component": provenance_observation,
        "host_to_device": transfer,
        "normal_forward": normal,
        "attention_forward": attention,
        "normal_attention_numerical_comparison": {
            "within_frozen_e5_cuda_tolerance": True,
            "rtol": CUDA_EQUIVALENCE_RTOL,
            "atol": CUDA_EQUIVALENCE_ATOL,
            "bitwise_identical": measured_bitwise_identical,
            "maximum_absolute_difference": measured_max_abs_difference,
            "note": (
                "CUDA/PyG scatter reductions are not required to be bitwise "
                "repeatable; this uses the frozen E.5 CUDA comparison policy"
            ),
        },
    }
    del attention_result, normal_result, model, cuda_data, data
    gc.collect()
    torch.cuda.empty_cache()
    return deterministic, observational


def run_real_pilot(
    project_root: Path,
    config: PilotConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the bounded real E1/E2/E3 CUDA pilot and return split evidence."""

    project_root = Path(project_root).resolve()
    sources_before = require_frozen_sources(project_root)
    observed_environment = environment_identity()
    deterministic_environment = {
        key: value
        for key, value in observed_environment.items()
        if key != "gpu_free_memory_at_pilot_start_bytes"
    }
    deterministic_tasks: list[dict[str, Any]] = []
    observational_tasks: list[dict[str, Any]] = []
    for task in TASKS:
        deterministic, observational = _measure_one_task(
            task, config, project_root
        )
        deterministic_tasks.append(deterministic)
        observational_tasks.append(observational)
    sources_after = require_frozen_sources(project_root)
    if sources_after != sources_before:
        raise ResourcePilotContractError("frozen sources changed during the pilot")
    all_contracts_pass = all(
        item["forward_contract"]["status"] == "PASS"
        and item["parameter_count"] == item["expected_parameter_count"]
        and item["d4_source_lazy_read_only"]
        and item["d6_batch_read_only"]
        for item in deterministic_tasks
    )
    deterministic_report = {
        "stage": "E.6",
        "pilot_version": PILOT_VERSION,
        "evidence_class": "deterministic_contract_facts",
        "environment": deterministic_environment,
        "configuration": {
            "batch_size": config.batch_size,
            "warmup_iterations_per_cuda_mode": config.warmup_iterations,
            "measured_iterations_per_mode": config.measured_iterations,
            "tasks": list(TASKS),
            "split": "train",
            "seed": SEED,
            "maximum_allowed_batch_size": MAX_BATCH_SIZE,
            "maximum_allowed_warmup_iterations": MAX_WARMUP_ITERATIONS,
            "maximum_allowed_measured_iterations": MAX_MEASURED_ITERATIONS,
            "sample_selection": "deterministic capture-order-spanning indices",
        },
        "frozen_sources": sources_after,
        "tasks": deterministic_tasks,
        "runtime_observations_path": (
            "results/stage_e6/htgnn_resource_pilot_observations.json"
        ),
        "runtime_timings_in_canonical_evidence": False,
        "frozen_architecture_mutated": False,
        "training_performed": False,
        "predictive_metrics_computed": False,
        "pruning_performed": False,
        "batch_size_optimization_performed": False,
        "bounded_cuda_forward_operational": all_contracts_pass,
        "verification": {"status": "NOT_RUN"},
        "overall_status": "PASS" if all_contracts_pass else "FAIL",
    }
    provenance_visible = any(
        item["e2_provenance_validation_component"]["appears_measurable"]
        for item in observational_tasks
    )
    observational_report = {
        "stage": "E.6",
        "pilot_version": PILOT_VERSION,
        "evidence_class": "runtime_performance_observations",
        "canonical_or_byte_reproducible": False,
        "hardware_specific": True,
        "environment": observed_environment,
        "configuration": deterministic_report["configuration"],
        "tasks": observational_tasks,
        "e2_provenance_overhead": {
            "appears_measurable_in_at_least_one_task": provenance_visible,
            "investigate_before_e7": provenance_visible,
            "recommendation_basis": (
                "measured frozen provenance-validation component relative to "
                "complete repeated E.2 conversion; no optimization was attempted"
            ),
        },
        "resource_or_correctness_blocker_observed_within_bound": False,
        "training_feasibility_claimed": False,
        "generalizable_beyond_this_environment": False,
    }
    return deterministic_report, observational_report
