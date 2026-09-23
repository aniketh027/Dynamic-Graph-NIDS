"""Build compact deterministic Stage E.5 complete-model validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch_geometric

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import adapter as d4  # noqa: E402
from src.graph import batching as d6  # noqa: E402
from src.graph import htgnn_classifier as e5  # noqa: E402
from src.graph import pyg_adapter as e2  # noqa: E402
from src.graph import schema as d2  # noqa: E402


SEED = 20260921
CUDA_ISOLATION_RTOL = 1e-5
CUDA_ISOLATION_ATOL = 1e-6
ISOLATION_MIN_PERTURBED_ABS_CHANGE = 1e-3
ISOLATION_ADVERSARIAL_SHIFT = 1e-3
CUDA_ISOLATION_STRESS_REPLAYS = 20
APPROVED_E4_REPORT_SHA256 = (
    "8de906e6adc61c6f0f32f7b4a2c2e1a2b32a39f5fd0b1c9f60f26ebe7e655db9"
)
FROZEN_COMMITS = {
    "E.2": "c63a969d5cbb602e9679880a39ddc7998ec6344c",
    "E.3": "95ecf6c4816e2ddee571b7252b4eec9b346eb889",
    "E.4": "0695ecefbfee8f66188caed4d571eea1f6eb2ca7",
}
FROZEN_TAGS = {
    "E.2": "v0.5-stage-e2-heterodata",
    "E.3": "v0.6-stage-e3-feature-encoder",
    "E.4": "v0.7-stage-e4-hierarchical-encoder",
}
EXPECTED_PARAMETERS = {"E1": 7183, "E2": 7183, "E3": 6721}
EXPECTED_SHAPES = {"E1": [3, 15], "E2": [3, 15], "E3": [3]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrity_threat_model(project_root: Path) -> dict[str, Any]:
    stage_document_path = project_root / "docs" / "stage_e5_htgnn_classifier.md"
    handoff_path = project_root / "PROJECT_HANDOFF.md"
    stage_document = stage_document_path.read_text(encoding="utf-8")
    handoff = handoff_path.read_text(encoding="utf-8")
    module_contract = e5.__doc__ or ""
    documents = (module_contract, stage_document, handoff)
    boundary_documented = all(
        "supported public model API" in text
        and "arbitrary" in text
        and "Python" in text
        and "__class__" in text
        and "_modules" in text
        and "same-process" in text
        for text in documents
    )
    audit_policy_documented = bool(
        "must not require resilience" in handoff
        and "does not authorize E.6" in handoff
        and "not E.5 acceptance requirements" in stage_document
    )
    return {
        "status": "PASS"
        if boundary_documented and audit_policy_documented
        else "FAIL",
        "required_scope": (
            "construction-time task/output immutability through supported public API"
        ),
        "supported_audit_checks": [
            "public task mutation rejection",
            "ordinary and currently detected head corruption",
            "task and output shapes",
            "trainability and state_dict structure",
            "sample-only readout and leakage independence",
            "exact E.4 attention passthrough and topology enforcement",
            "multi-graph isolation and nine-pair integration",
            "frozen-data integrity",
        ],
        "outside_supported_api": [
            "__class__ reassignment",
            "direct nn.Module._modules rewriting",
            "method or validation monkeypatching",
            "module-global or constant rebinding",
            "class-dictionary mutation",
            "imported module or function replacement",
            "equivalent arbitrary in-process Python mutation",
        ],
        "arbitrary_same_process_python_execution_in_scope": False,
        "reflective_tampering_survival_is_acceptance_gate": False,
        "isolated_internal_head_detection_is_defense_in_depth": True,
        "architecture_leakage_topology_and_integrity_checks_unchanged": True,
        "stage_document_path": "docs/stage_e5_htgnn_classifier.md",
        "stage_document_sha256": _sha256(stage_document_path),
        "handoff_audit_policy_documented": audit_policy_documented,
    }


def _load_approved_e4_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_e4" / "hierarchical_encoder_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied_hash != APPROVED_E4_REPORT_SHA256:
        raise ValueError("E.4 report does not have the approved canonical hash")
    if d2.canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("E.4 report canonical content is invalid")
    if report.get("overall_status") != "PASS":
        raise ValueError("E.4 report is not passing")
    return report


def _device_check(data: Any, device: str) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for task in ("E1", "E2", "E3"):
        torch.manual_seed(SEED)
        moved = data.clone().to(device)
        model = e5.HTGNNClassifier(task).to(device)
        logits = model(moved)
        logits.sum().backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        expected_shape = (3,) if task == "E3" else (3, 15)
        task_results[task] = {
            "logit_shape": list(logits.shape),
            "expected_shape": list(expected_shape),
            "finite_logits": bool(torch.isfinite(logits).all()),
            "all_gradients_present": all(item is not None for item in gradients),
            "finite_gradients": all(
                item is not None and bool(torch.isfinite(item).all())
                for item in gradients
            ),
            "backward_scalar": "logits.sum()",
        }
    if device == "cuda":
        torch.cuda.synchronize()
    passed = all(
        item["logit_shape"] == item["expected_shape"]
        and item["finite_logits"]
        and item["all_gradients_present"]
        and item["finite_gradients"]
        for item in task_results.values()
    )
    return {"status": "PASS" if passed else "FAIL", "tasks": task_results}


class _MalformedFifteenOutputHead(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :15]


def _task_and_head_contract_checks(data: Any) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for task in ("E1", "E2", "E3"):
        model = e5.HTGNNClassifier(task)
        normal_task_rejected = False
        normal_private_task_rejected = False
        try:
            model.task = "E3"  # type: ignore[misc]
        except AttributeError:
            normal_task_rejected = True
        try:
            model._task = "E3"  # type: ignore[attr-defined]
        except AttributeError:
            normal_private_task_rejected = True
        low_level_public_task_rejected = False
        try:
            object.__setattr__(model, "task", "E3")
        except AttributeError:
            low_level_public_task_rejected = True
        object.__setattr__(model, "_task", "E3")
        object.__setattr__(model, "_task_id", "E3")
        object.__setattr__(model, "_expected_output_width", 1)
        object.__setattr__(model, "shadow_task", "E3")
        model.__dict__["task"] = "E3"
        task_results[task] = {
            "concrete_model_type": type(model).__name__,
            "normal_task_assignment_rejected": normal_task_rejected,
            "normal_private_task_assignment_rejected": normal_private_task_rejected,
            "low_level_public_task_assignment_rejected": low_level_public_task_rejected,
            "low_level_instance_injection_ineffective": model.task == task,
            "authoritative_task_after_attacks": model.task,
            "authorized_head_width_after_attacks": model.head.out_features,
        }

    normal_head_cases: dict[str, Any] = {}
    for name, task, replacement in (
        ("e1_wrong_width", "E1", torch.nn.Linear(32, 1)),
        ("e3_wrong_width", "E3", torch.nn.Linear(32, 15)),
        ("e1_wrong_input_width", "E1", torch.nn.Linear(16, 15)),
        ("e1_custom_module", "E1", _MalformedFifteenOutputHead()),
    ):
        model = e5.HTGNNClassifier(task)
        original = model.head
        error = ""
        try:
            model.head = replacement
        except e5.TaskHeadContractError as exc:
            error = str(exc)
        normal_head_cases[name] = {
            "assignment_rejected": bool(error),
            "original_head_retained": model.head is original,
            "error": error,
        }

    low_level_cases: dict[str, Any] = {}
    for name, task, replacement, bypass in (
        ("e1_wrong_width", "E1", torch.nn.Linear(32, 1), "object_setattr"),
        ("e3_wrong_width", "E3", torch.nn.Linear(32, 15), "object_setattr"),
        ("e1_wrong_input_width", "E1", torch.nn.Linear(16, 15), "object_setattr"),
        ("e1_custom_module", "E1", _MalformedFifteenOutputHead(), "object_setattr"),
        ("e1_registered_module", "E1", torch.nn.Linear(32, 1), "modules_dict"),
    ):
        model = e5.HTGNNClassifier(task).eval()
        calls: list[str] = []
        handles = [
            model.feature_encoder.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("E3")
            ),
            model.graph_encoder.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("E4")
            ),
        ]
        if bypass == "object_setattr":
            object.__setattr__(model, "head", replacement)
        else:
            model._modules["head"] = replacement
        error = ""
        try:
            with torch.no_grad():
                model(data)
        except e5.TaskHeadContractError as exc:
            error = str(exc)
        finally:
            for handle in handles:
                handle.remove()
        low_level_cases[name] = {
            "bypass": bypass,
            "forward_rejected": bool(error),
            "rejected_before_e3_or_e4": calls == [],
            "error": error,
        }

    for name in ("weight_shape", "bias_shape", "missing_bias"):
        model = e5.HTGNNClassifier("E1").eval()
        if name == "weight_shape":
            model.head.weight = torch.nn.Parameter(torch.randn(14, 32))
        elif name == "bias_shape":
            model.head.bias = torch.nn.Parameter(torch.randn(14))
        else:
            model.head.bias = None
        calls: list[str] = []
        handle = model.feature_encoder.register_forward_pre_hook(
            lambda _module, _inputs: calls.append("E3")
        )
        error = ""
        try:
            with torch.no_grad():
                model(data)
        except e5.TaskHeadContractError as exc:
            error = str(exc)
        finally:
            handle.remove()
        low_level_cases[f"e1_{name}"] = {
            "bypass": "parameter_structure_mutation",
            "forward_rejected": bool(error),
            "rejected_before_e3_or_e4": calls == [],
            "error": error,
        }

    combined = e5.HTGNNClassifier("E1").eval()
    object.__setattr__(combined, "_task", "E3")
    object.__setattr__(combined, "task_id", "E3")
    object.__setattr__(combined, "head", torch.nn.Linear(32, 1))
    combined_error = ""
    try:
        with torch.no_grad():
            combined(data)
    except e5.TaskHeadContractError as exc:
        combined_error = str(exc)

    state_dicts: dict[str, Any] = {}
    for task in ("E1", "E2", "E3"):
        first = e5.HTGNNClassifier(task)
        second = e5.HTGNNClassifier(task)
        first_state = first.state_dict()
        second_state = second.state_dict()
        state_dicts[task] = {
            "deterministic_key_structure": tuple(first_state) == tuple(second_state),
            "encoder_keys_present": any(
                key.startswith("feature_encoder.") for key in first_state
            )
            and any(key.startswith("graph_encoder.") for key in first_state),
            "head_weight_shape": list(first_state["head.weight"].shape),
            "head_bias_shape": list(first_state["head.bias"].shape),
        }

    passed = bool(
        all(
            item["normal_task_assignment_rejected"]
            and item["normal_private_task_assignment_rejected"]
            and item["low_level_public_task_assignment_rejected"]
            and item["low_level_instance_injection_ineffective"]
            and item["authoritative_task_after_attacks"] == task
            and item["authorized_head_width_after_attacks"]
            == (1 if task == "E3" else 15)
            for task, item in task_results.items()
        )
        and all(
            item["assignment_rejected"] and item["original_head_retained"]
            for item in normal_head_cases.values()
        )
        and all(
            item["forward_rejected"] and item["rejected_before_e3_or_e4"]
            for item in low_level_cases.values()
        )
        and combined.task == "E1"
        and bool(combined_error)
        and all(
            item["deterministic_key_structure"]
            and item["encoder_keys_present"]
            for item in state_dicts.values()
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "authoritative_mechanism": (
            "exact concrete task-specific model type resolved through a "
            "module-level MappingProxyType contract; no instance task field"
        ),
        "task_identity_attacks": task_results,
        "normal_head_replacement": normal_head_cases,
        "low_level_head_corruption": low_level_cases,
        "combined_e1_to_e3_attack": {
            "authoritative_task": combined.task,
            "forward_rejected": bool(combined_error),
            "error": combined_error,
        },
        "runtime_validation": [
            "registered resolved module identity",
            "exact nn.Linear type",
            "in_features",
            "task-authorized out_features",
            "weight Parameter and shape",
            "bias Parameter and shape",
            "returned logit shape",
        ],
        "state_dict_contract": state_dicts,
    }


def _complete_model_topology_checks(data: Any) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name in (
        "feature_semantic_redirect",
        "feature_edge_reorder",
        "cross_graph_category_to_sample_redirect",
    ):
        mutated = data.clone()
        if name == "feature_semantic_redirect":
            destinations = mutated[e2.FEATURE_RELATION].edge_index[1]
            destinations[0] = (int(destinations[0]) + 1) % 8
        elif name == "feature_edge_reorder":
            edges = mutated[e2.FEATURE_RELATION].edge_index
            mutated[e2.FEATURE_RELATION].edge_index = edges.flip(1)
        else:
            mutated[e2.CATEGORY_RELATION].edge_index[1, 0] = 1

        model = e5.HTGNNClassifier("E1").eval()
        calls: list[str] = []
        handles = [
            model.graph_encoder.feature_to_category.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("feature_to_category")
            ),
            model.graph_encoder.category_to_sample.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("category_to_sample")
            ),
        ]
        error = ""
        try:
            with torch.no_grad():
                model(mutated)
        except ValueError as exc:
            error = str(exc)
        finally:
            for handle in handles:
                handle.remove()
        cases[name] = {
            "rejected": bool(error),
            "rejected_before_message_passing": calls == [],
            "error": error,
        }
    passed = all(
        item["rejected"] and item["rejected_before_message_passing"]
        for item in cases.values()
    )
    return {"status": "PASS" if passed else "FAIL", "cases": cases}


def _multi_graph_logit_isolation(data: Any) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = e5.HTGNNClassifier("E1").eval()
    original = data.clone()
    changed = data.clone()
    changed["feature"].x[:78] += 1000.0
    with torch.no_grad():
        original_logits = model(original)
        changed_logits = model(changed)
    cpu_graph_zero_material = bool(
        torch.max(torch.abs(original_logits[0] - changed_logits[0]))
        > ISOLATION_MIN_PERTURBED_ABS_CHANGE
    )
    cpu_untouched_bitwise = bool(
        torch.equal(original_logits[1:], changed_logits[1:])
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Stage E.1 requires CUDA, but CUDA is unavailable")
    cuda_graph_zero_material: list[bool] = []
    cuda_untouched_close: list[bool] = []
    cuda_adversarial_shift_rejected: list[bool] = []
    for _replay in range(CUDA_ISOLATION_STRESS_REPLAYS):
        torch.manual_seed(SEED)
        cuda_original = data.clone().to("cuda")
        cuda_changed = cuda_original.clone()
        cuda_changed["feature"].x[:78] += 1000.0
        cuda_model = e5.HTGNNClassifier("E1").to("cuda").eval()
        with torch.no_grad():
            cuda_original_logits = cuda_model(cuda_original)
            cuda_changed_logits = cuda_model(cuda_changed)
        torch.cuda.synchronize()
        cuda_graph_zero_material.append(
            bool(
                torch.max(
                    torch.abs(cuda_original_logits[0] - cuda_changed_logits[0])
                )
                > ISOLATION_MIN_PERTURBED_ABS_CHANGE
            )
        )
        close_passed = True
        try:
            torch.testing.assert_close(
                cuda_original_logits[1:],
                cuda_changed_logits[1:],
                rtol=CUDA_ISOLATION_RTOL,
                atol=CUDA_ISOLATION_ATOL,
            )
        except AssertionError:
            close_passed = False
        cuda_untouched_close.append(close_passed)
        sanity_rejected = False
        try:
            torch.testing.assert_close(
                cuda_original_logits[1:],
                cuda_changed_logits[1:] + ISOLATION_ADVERSARIAL_SHIFT,
                rtol=CUDA_ISOLATION_RTOL,
                atol=CUDA_ISOLATION_ATOL,
            )
        except AssertionError:
            sanity_rejected = True
        cuda_adversarial_shift_rejected.append(sanity_rejected)

    cuda_all_passed = bool(
        all(cuda_graph_zero_material)
        and all(cuda_untouched_close)
        and all(cuda_adversarial_shift_rejected)
    )
    return {
        "status": "PASS"
        if cpu_graph_zero_material and cpu_untouched_bitwise and cuda_all_passed
        else "FAIL",
        "batch_size": int(original_logits.shape[0]),
        "only_graph_zero_numeric_features_perturbed": True,
        "comparison_dtype": "torch.float32",
        "graph_zero_minimum_required_max_abs_change": (
            ISOLATION_MIN_PERTURBED_ABS_CHANGE
        ),
        "graph_zero_material_response_all": bool(
            cpu_graph_zero_material and all(cuda_graph_zero_material)
        ),
        "untouched_graphs_compared_per_graph": True,
        "cpu": {
            "comparison": "bitwise equality",
            "bitwise_identical": cpu_untouched_bitwise,
        },
        "cuda": {
            "comparison": "torch.testing.assert_close per graph",
            "rtol": CUDA_ISOLATION_RTOL,
            "atol": CUDA_ISOLATION_ATOL,
            "stress_replay_count": CUDA_ISOLATION_STRESS_REPLAYS,
            "all_stress_replays_passed": cuda_all_passed,
            "adversarial_shift": ISOLATION_ADVERSARIAL_SHIFT,
            "adversarial_shift_rejected_all": bool(
                all(cuda_adversarial_shift_rejected)
            ),
            "raw_nondeterministic_deltas_stored": False,
        },
    }


def _run_pytest(project_root: Path, args: list[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *args]
    completed = subprocess.run(
        command,
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


def _write_report(report: dict[str, Any], output: Path) -> None:
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    report["report_sha256"] = d2.canonical_sha256(unhashed)
    report_bytes = d2.canonical_json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(report_bytes)
    physical_hash = hashlib.sha256(report_bytes).hexdigest()
    output.with_suffix(".sha256").write_text(
        f"{physical_hash}  {output.name}\n", encoding="ascii"
    )


def build_report(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    e4_report = _load_approved_e4_report(project_root)
    integrity_threat_model = _integrity_threat_model(project_root)
    torch.manual_seed(SEED)
    models = {task: e5.HTGNNClassifier(task).eval() for task in ("E1", "E2", "E3")}
    parameter_counts = {
        task: {
            "expected": EXPECTED_PARAMETERS[task],
            "measured": sum(parameter.numel() for parameter in model.parameters()),
            "shared_e3_e4_encoder": sum(
                parameter.numel()
                for module in (model.feature_encoder, model.graph_encoder)
                for parameter in module.parameters()
            ),
            "head": sum(parameter.numel() for parameter in model.head.parameters()),
            "difference": sum(parameter.numel() for parameter in model.parameters())
            - EXPECTED_PARAMETERS[task],
        }
        for task, model in models.items()
    }

    parameter_id_sets = {
        task: {id(parameter) for parameter in model.parameters()}
        for task, model in models.items()
    }
    task_isolation = {
        "separate_instances": len({id(model) for model in models.values()}) == 3,
        "no_shared_parameters": all(
            parameter_id_sets[left].isdisjoint(parameter_id_sets[right])
            for left, right in (("E1", "E2"), ("E1", "E3"), ("E2", "E3"))
        ),
        "head_outputs": {
            task: model.head.out_features for task, model in models.items()
        },
        "automatic_weight_transfer": False,
    }

    records: list[dict[str, Any]] = []
    paired_data: dict[tuple[str, str], Any] = {}
    all_real_valid = True
    all_unmodified = True
    processed_root = project_root / "data" / "processed" / "preprocessed"
    first_data = None
    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(
                experiment, split, processed_root=processed_root
            )
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            batch = d6.build_logical_batch(dataset, indices)
            batch_before = np.array(batch.feature_x, copy=True)
            data = e2.logical_batch_to_heterodata(batch, project_root=project_root)
            paired_data[(experiment, split)] = data
            x_before = data["feature"].x.clone()
            y_before = data["sample"].y.clone()
            feature_edges_before = data[e2.FEATURE_RELATION].edge_index.clone()
            category_edges_before = data[e2.CATEGORY_RELATION].edge_index.clone()
            with torch.no_grad():
                normal = models[experiment](data)
                attention_logits, attention = models[experiment](
                    data, return_attention=True
                )
            expected_shape = EXPECTED_SHAPES[experiment]
            feature_edges, feature_alpha = attention[e2.FEATURE_RELATION]
            category_edges, category_alpha = attention[e2.CATEGORY_RELATION]
            topology_unchanged = bool(
                torch.equal(feature_edges, feature_edges_before)
                and torch.equal(category_edges, category_edges_before)
                and feature_alpha.shape == (3 * 78, 4)
                and category_alpha.shape == (3 * 8, 4)
            )
            source_unmodified = bool(
                np.array_equal(batch.feature_x, batch_before)
                and torch.equal(data["feature"].x, x_before)
                and torch.equal(data["sample"].y, y_before)
                and torch.equal(
                    data[e2.FEATURE_RELATION].edge_index, feature_edges_before
                )
                and torch.equal(
                    data[e2.CATEGORY_RELATION].edge_index, category_edges_before
                )
                and dataset.is_lazy_read_only
            )
            valid = bool(
                list(normal.shape) == expected_shape
                and torch.isfinite(normal).all()
                and torch.equal(normal, attention_logits)
                and topology_unchanged
            )
            all_real_valid &= valid
            all_unmodified &= source_unmodified
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "matching_model_task": models[experiment].task,
                    "source_dataset_length": len(dataset),
                    "tested_indices": indices,
                    "batch_size": batch.batch_size,
                    "logit_shape": list(normal.shape),
                    "finite_logits": bool(torch.isfinite(normal).all()),
                    "attention_mode_logits_identical": bool(
                        torch.equal(normal, attention_logits)
                    ),
                    "feature_attention_shape": list(feature_alpha.shape),
                    "category_attention_shape": list(category_alpha.shape),
                    "topology_unchanged": topology_unchanged,
                    "source_inputs_unmodified": source_unmodified,
                }
            )
            if first_data is None:
                first_data = data

    if first_data is None:
        raise RuntimeError("no real integration batch was produced")

    task_head_contract = _task_and_head_contract_checks(first_data)
    topology_corruption = _complete_model_topology_checks(first_data)
    multi_graph_isolation = _multi_graph_logit_isolation(first_data)

    readout_capture: dict[str, torch.Tensor] = {}
    graph_handle = models["E1"].graph_encoder.register_forward_hook(
        lambda _module, _inputs, output: readout_capture.__setitem__(
            "sample_embedding", output.detach().clone()
        )
    )
    head_handle = models["E1"].head.register_forward_pre_hook(
        lambda _module, inputs: readout_capture.__setitem__(
            "head_input", inputs[0].detach().clone()
        )
    )
    with torch.no_grad():
        models["E1"](first_data)
    graph_handle.remove()
    head_handle.remove()
    readout_path = {
        "sample_embedding_only": bool(
            readout_capture["sample_embedding"].shape == (3, 32)
            and torch.equal(
                readout_capture["sample_embedding"], readout_capture["head_input"]
            )
        ),
        "classifier_input_shape": list(readout_capture["head_input"].shape),
        "forbidden_bypasses_absent": [
            "raw feature.x",
            "flattened 5x78 features",
            "feature embeddings directly",
            "pooled feature nodes",
            "pooled category nodes",
            "targets",
            "metadata",
            "sample indices",
            "experiment or split identity",
        ],
    }

    leakage_data = first_data.clone()
    with torch.no_grad():
        baseline = models["E1"](leakage_data)
        leakage_data["sample"].y = leakage_data["sample"].y + 1000
        leakage_data["sample"].source_sample_indices = torch.tensor([99, 98, 97])
        leakage_data.e2_audit = {
            "experiment": "E3",
            "split": "changed",
            "paths": ["changed"],
            "provenance": {"hash": "changed"},
        }
        changed = models["E1"](leakage_data)
        del leakage_data["sample"].y
        del leakage_data["sample"].source_sample_indices
        del leakage_data.e2_audit
        absent = models["E1"](leakage_data)
    target_and_metadata_independence = bool(
        torch.equal(baseline, changed) and torch.equal(baseline, absent)
    )

    with torch.no_grad():
        normal_logits = models["E1"](first_data)
        attention_logits, attention = models["E1"](
            first_data, return_attention=True
        )
    attention_mode_equivalence = bool(torch.equal(normal_logits, attention_logits))
    attention_contract = {
        "exact_e4_payload_pass_through": bool(
            torch.equal(
                attention[e2.FEATURE_RELATION][0],
                first_data[e2.FEATURE_RELATION].edge_index,
            )
            and torch.equal(
                attention[e2.CATEGORY_RELATION][0],
                first_data[e2.CATEGORY_RELATION].edge_index,
            )
        ),
        "raw_four_heads_not_averaged": bool(
            attention[e2.FEATURE_RELATION][1].shape == (234, 4)
            and attention[e2.CATEGORY_RELATION][1].shape == (24, 4)
        ),
        "no_added_or_self_loop_topology": bool(
            not models["E1"].graph_encoder.feature_to_category.add_self_loops
            and not models["E1"].graph_encoder.category_to_sample.add_self_loops
        ),
    }

    e1_e3_results: list[dict[str, Any]] = []
    equivalence_model = models["E1"]
    for split in d4.SPLIT_STEMS:
        e1_data = paired_data[("E1", split)]
        e3_data = paired_data[("E3", split)]
        with torch.no_grad():
            e1_features = equivalence_model.feature_encoder(
                e1_data["feature"].x, e1_data["feature"].feature_id
            )
            e3_features = equivalence_model.feature_encoder(
                e3_data["feature"].x, e3_data["feature"].feature_id
            )
            e1_samples = equivalence_model._encode_sample_embeddings(e1_data)
            e3_samples = equivalence_model._encode_sample_embeddings(e3_data)
        e1_e3_results.append(
            {
                "split": split,
                "same_encoder_instance": True,
                "e3_feature_embeddings_identical": bool(
                    torch.equal(e1_features, e3_features)
                ),
                "e4_sample_embeddings_identical": bool(
                    torch.equal(e1_samples, e3_samples)
                ),
            }
        )
    e1_e3_all = all(
        item["e3_feature_embeddings_identical"]
        and item["e4_sample_embeddings_identical"]
        for item in e1_e3_results
    )

    cpu = _device_check(first_data, "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage E.1 requires CUDA, but CUDA is unavailable")
    cuda = _device_check(first_data, "cuda")
    torch.manual_seed(SEED)
    trainability_model = e5.HTGNNClassifier("E1")
    all_require_grad = all(
        parameter.requires_grad for parameter in trainability_model.parameters()
    )
    trainability_logits = trainability_model(first_data)
    trainability_logits.sum().backward()
    all_receive_gradients = all(
        parameter.grad is not None for parameter in trainability_model.parameters()
    )
    head_before = trainability_model.head.weight.detach().clone()
    with torch.no_grad():
        trainability_model.head.weight.add_(
            0.01 * trainability_model.head.weight.grad
        )
    trainability = {
        "status": "PASS"
        if all_require_grad
        and all_receive_gradients
        and not torch.equal(head_before, trainability_model.head.weight)
        else "FAIL",
        "all_parameters_require_grad": all_require_grad,
        "all_parameters_received_gradients": all_receive_gradients,
        "normal_in_place_parameter_update_changed_weight": bool(
            not torch.equal(head_before, trainability_model.head.weight)
        ),
        "architecture_protection_does_not_freeze_numerical_parameters": True,
    }
    integrity = d4.verify_frozen_integrity(project_root)
    parameter_contract_pass = all(
        item["measured"] == item["expected"]
        and item["shared_e3_e4_encoder"] == 6688
        for item in parameter_counts.values()
    )
    task_isolation_pass = bool(
        task_isolation["separate_instances"]
        and task_isolation["no_shared_parameters"]
        and task_isolation["head_outputs"] == {"E1": 15, "E2": 15, "E3": 1}
    )
    semantic_pass = all(
        (
            parameter_contract_pass,
            task_isolation_pass,
            task_head_contract["status"] == "PASS",
            integrity_threat_model["status"] == "PASS",
            topology_corruption["status"] == "PASS",
            multi_graph_isolation["status"] == "PASS",
            readout_path["sample_embedding_only"],
            target_and_metadata_independence,
            attention_mode_equivalence,
            attention_contract["exact_e4_payload_pass_through"],
            attention_contract["raw_four_heads_not_averaged"],
            attention_contract["no_added_or_self_loop_topology"],
            all_real_valid,
            all_unmodified,
            e1_e3_all,
            cpu["status"] == "PASS",
            cuda["status"] == "PASS",
            trainability["status"] == "PASS",
            integrity["overall_status"] == "PASS",
        )
    )

    return {
        "stage": "E.5",
        "model_version": e5.MODEL_VERSION,
        "implementation": {
            "source_path": "src/graph/htgnn_classifier.py",
            "source_sha256": _sha256(
                project_root / "src" / "graph" / "htgnn_classifier.py"
            ),
        },
        "frozen_upstream": {
            "commits": FROZEN_COMMITS,
            "tags": FROZEN_TAGS,
            "e4_report_sha256": e4_report["report_sha256"],
            "e3_report_sha256": e4_report["upstream"]["e3_report_sha256"],
            "e2_report_sha256": e4_report["upstream"]["e2_report_sha256"],
            "e3_source_sha256": _sha256(
                project_root / "src" / "graph" / "feature_encoder.py"
            ),
            "e4_source_sha256": _sha256(
                project_root / "src" / "graph" / "hierarchical_encoder.py"
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "api": {
            "E1": "HTGNNClassifier('E1')(HeteroData) -> raw logits [B,15]",
            "E2": "HTGNNClassifier('E2')(HeteroData) -> raw logits [B,15]",
            "E3": "HTGNNClassifier('E3')(HeteroData) -> raw binary logits [B]",
            "attention": "forward(data, return_attention=True) -> (raw logits, exact E.4 attention payload)",
            "production_activation": None,
        },
        "parameter_counts": parameter_counts,
        "task_isolation": task_isolation,
        "construction_time_task_and_head_contract": task_head_contract,
        "integrity_threat_model": integrity_threat_model,
        "complete_model_topology_corruption": topology_corruption,
        "multi_graph_logit_isolation": multi_graph_isolation,
        "numerical_parameter_trainability": trainability,
        "readout_path": readout_path,
        "target_and_metadata_independence": target_and_metadata_independence,
        "attention_mode_equivalence": attention_mode_equivalence,
        "attention_contract": attention_contract,
        "e1_e3_same_encoder_pre_head_results": e1_e3_results,
        "e1_e3_pre_head_equivalence_all": e1_e3_all,
        "cpu_forward_backward": cpu,
        "cuda_forward_backward": cuda,
        "real_nine_pair_batches": records,
        "real_d4_d6_e2_e3_e4_e5_integration_all": all_real_valid,
        "source_inputs_unmodified_all": all_unmodified,
        "stage_c_integrity": {
            "status": "PASS" if integrity["stage_c_sha256_matches"] else "FAIL",
            "matched": integrity["stage_c_artifact_count"],
            "expected": integrity["stage_c_expected_count"],
        },
        "raw_data_integrity": {
            "status": "PASS" if integrity["raw_csv_sha256_matches"] else "FAIL",
            "matched": integrity["raw_csv_count"],
            "expected": integrity["raw_csv_expected_count"],
        },
        "verification": {"status": "NOT_RUN"},
        "excluded": [
            "softmax or sigmoid",
            "loss functions",
            "class weights or pos_weight",
            "optimizer or scheduler",
            "DataLoader or epoch loop",
            "early stopping or checkpoint selection",
            "metrics or threshold selection",
            "training",
            "pruning",
            "model weights",
            "E.6 and later stages",
        ],
        "untrained_logits_are_meaningful_predictions": False,
        "overall_status": "PASS" if semantic_pass else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="run and record the E.5 focused, frozen regression, and full suites",
    )
    args = parser.parse_args()
    output = args.output or (
        args.project_root
        / "results"
        / "stage_e5"
        / "htgnn_classifier_validation.json"
    )
    report = build_report(args.project_root)
    _write_report(report, output)

    if args.run_tests:
        verification = {
            "focused_e5": _run_pytest(
                args.project_root, ["tests/test_htgnn_classifier.py"]
            ),
            "frozen_e2_e4_regression": _run_pytest(
                args.project_root,
                [
                    "tests/test_pyg_adapter.py",
                    "tests/test_feature_encoder.py",
                    "tests/test_hierarchical_encoder.py",
                ],
            ),
            "complete_project_suite": _run_pytest(args.project_root, []),
        }
        verification["status"] = (
            "PASS"
            if all(item["status"] == "PASS" for item in verification.values())
            else "FAIL"
        )
        report["verification"] = verification
        report["overall_status"] = (
            "PASS"
            if report["overall_status"] == "PASS"
            and verification["status"] == "PASS"
            else "FAIL"
        )
        _write_report(report, output)

    physical_hash = _sha256(output)
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"physical_sha256={physical_hash}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
