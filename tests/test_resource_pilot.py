from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.graph import pyg_adapter as e2
from src.graph import resource_pilot as e6
from src.graph import schema as d2
from tests.test_htgnn_classifier import _synthetic_data


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_REPORT = (
    ROOT / "results" / "stage_e6" / "htgnn_resource_pilot_contract.json"
)
CONTRACT_SHA = CONTRACT_REPORT.with_suffix(".sha256")
OBSERVATIONS = (
    ROOT / "results" / "stage_e6" / "htgnn_resource_pilot_observations.json"
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", e6.MAX_BATCH_SIZE + 1),
        ("warmup_iterations", 0),
        ("warmup_iterations", e6.MAX_WARMUP_ITERATIONS + 1),
        ("measured_iterations", 1),
        ("measured_iterations", e6.MAX_MEASURED_ITERATIONS + 1),
    ],
)
def test_pilot_config_enforces_resource_bounds(field: str, value: int) -> None:
    values = {
        "batch_size": 3,
        "warmup_iterations": 1,
        "measured_iterations": 2,
    }
    values[field] = value
    with pytest.raises(ValueError):
        e6.PilotConfig(**values)


def test_bounded_indices_are_unique_ordered_and_do_not_enumerate_split() -> None:
    indices = e6.bounded_sample_indices(1_000_000, 32)
    assert len(indices) == 32
    assert len(set(indices)) == 32
    assert indices == tuple(sorted(indices))
    assert indices[0] == 0
    assert indices[-1] == 999_999


def test_real_batch_builder_requests_only_bounded_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dataset:
        is_lazy_read_only = True

        def __len__(self) -> int:
            return 1000

        def __iter__(self):
            raise AssertionError("full-split iteration is forbidden")

    dataset = Dataset()
    captured: dict[str, object] = {}
    monkeypatch.setattr(e6.d4, "LazyGraphDataset", lambda *args, **kwargs: dataset)

    def fake_build(source: object, indices: tuple[int, ...]) -> object:
        captured["source"] = source
        captured["indices"] = indices
        return SimpleNamespace(batch_size=len(indices))

    monkeypatch.setattr(e6.d6, "build_logical_batch", fake_build)
    config = e6.PilotConfig(batch_size=4, warmup_iterations=1, measured_iterations=2)
    returned_dataset, _batch, indices = e6.build_bounded_real_train_batch(
        "E1", config, ROOT
    )
    assert returned_dataset is dataset
    assert captured == {"source": dataset, "indices": indices}
    assert len(indices) == 4


@pytest.mark.parametrize(
    ("task", "expected_shape", "parameter_count"),
    [("E1", [3, 15], 7183), ("E2", [3, 15], 7183), ("E3", [3], 6721)],
)
def test_task_model_shapes_attention_equality_and_source_immutability(
    task: str, expected_shape: list[int], parameter_count: int
) -> None:
    torch.manual_seed(e6.SEED)
    data = _synthetic_data(3)
    model = e6.create_frozen_model(task)
    facts = e6.verify_forward_contract(task, model, data)
    assert facts["status"] == "PASS"
    assert facts["logit_shape"] == expected_shape
    assert facts["finite_logits"] is True
    assert facts["attention_logits_identical"] is True
    assert facts["attention_logits_bitwise_identical_observation"] is True
    assert facts["feature_attention_shape"] == [3 * 78, 4]
    assert facts["category_attention_shape"] == [3 * 8, 4]
    assert facts["feature_topology_identical"] is True
    assert facts["category_topology_identical"] is True
    assert facts["source_tensors_unchanged"] is True
    assert facts["model_state_unchanged"] is True
    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count


def test_task_model_selection_rejects_invalid_or_mismatched_task() -> None:
    with pytest.raises(ValueError):
        e6.create_frozen_model("E4")
    with pytest.raises(e6.ResourcePilotContractError):
        e6.verify_forward_contract(
            "E2", e6.create_frozen_model("E1"), _synthetic_data(1)
        )


class _FakeCudaRuntime:
    def __init__(self) -> None:
        self.synchronizations = 0
        self.resets = 0

    def synchronize(self) -> None:
        self.synchronizations += 1

    def reset_peak_memory_stats(self) -> None:
        self.resets += 1

    def memory_allocated(self) -> int:
        return 100

    def memory_reserved(self) -> int:
        return 200

    def max_memory_allocated(self) -> int:
        return 160

    def max_memory_reserved(self) -> int:
        return 280


def test_cuda_timing_synchronizes_separates_prepare_and_resets_memory() -> None:
    runtime = _FakeCudaRuntime()
    clock = iter([0, 1_000_000, 2_000_000, 5_000_000])
    prepared: list[str] = []

    def prepare() -> str:
        prepared.append("prepared")
        return prepared[-1]

    result, observation = e6.measure_cuda_callable(
        lambda value: value,
        prepare=prepare,
        warmup_iterations=1,
        measured_iterations=2,
        cuda_runtime=runtime,
        clock_ns=lambda: next(clock),
    )
    assert result == "prepared"
    assert len(prepared) == 3
    assert runtime.resets == 1
    assert runtime.synchronizations == 6
    assert observation["timing"]["samples"] == [1.0, 3.0]
    assert observation["peak_allocated_delta_bytes"] == 60
    assert observation["peak_reserved_delta_bytes"] == 80


def test_measurement_modes_are_explicit_and_non_overlapping() -> None:
    assert e6.MEASUREMENT_MODES == (
        "e2_conversion",
        "host_to_device",
        "normal_forward",
        "attention_forward",
    )
    assert len(set(e6.MEASUREMENT_MODES)) == 4


def test_e6_source_has_no_training_pruning_or_predictive_metric_components() -> None:
    paths = [
        ROOT / "src" / "graph" / "resource_pilot.py",
        ROOT / "scripts" / "run_stage_e6_resource_pilot.py",
    ]
    forbidden = {
        "CrossEntropyLoss",
        "BCEWithLogitsLoss",
        "AdamW",
        "Optimizer",
        "DataLoader",
        "backward",
        "train",
        "accuracy_score",
        "f1_score",
        "average_precision_score",
        "confusion_matrix",
        "prune",
    }
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert not (forbidden & names)


def test_frozen_e2_e5_sources_match_approved_hashes() -> None:
    assert e6.require_frozen_sources(ROOT) == e6.FROZEN_SOURCE_SHA256


def test_saved_reports_keep_contract_facts_separate_from_observations() -> None:
    if not CONTRACT_REPORT.exists() or not OBSERVATIONS.exists():
        pytest.skip("real E.6 pilot evidence has not been generated yet")
    contract_bytes = CONTRACT_REPORT.read_bytes()
    contract = json.loads(contract_bytes)
    supplied = contract.pop("report_sha256")
    assert d2.canonical_sha256(contract) == supplied
    contract["report_sha256"] = supplied
    assert d2.canonical_json_bytes(contract) == contract_bytes
    assert contract["overall_status"] == "PASS"
    assert contract["runtime_timings_in_canonical_evidence"] is False
    assert contract["training_performed"] is False
    assert contract["predictive_metrics_computed"] is False
    assert contract["pruning_performed"] is False
    assert contract["batch_size_optimization_performed"] is False
    assert len(contract["tasks"]) == 3
    observations = json.loads(OBSERVATIONS.read_text(encoding="utf-8"))
    assert observations["evidence_class"] == "runtime_performance_observations"
    assert observations["canonical_or_byte_reproducible"] is False
    assert observations["training_feasibility_claimed"] is False
    for task in observations["tasks"]:
        assert set(e6.MEASUREMENT_MODES).issubset(task)
    physical = hashlib.sha256(contract_bytes).hexdigest()
    assert CONTRACT_SHA.read_text(encoding="ascii") == (
        f"{physical}  {CONTRACT_REPORT.name}\n"
    )
