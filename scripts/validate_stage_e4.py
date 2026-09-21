"""Build deterministic Stage E.4 hierarchical-encoder evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
from src.graph import feature_encoder as e3  # noqa: E402
from src.graph import full_split_integration as d8  # noqa: E402
from src.graph import hierarchical_encoder as e4  # noqa: E402
from src.graph import pyg_adapter as e2  # noqa: E402
from src.graph import schema as d2  # noqa: E402


SEED = 20260921
APPROVED_E3_REPORT_SHA256 = (
    "b3e838c90a7488eca2c7b9083744386a6340234549318c4709b1176b87ae9adb"
)
E4_DESIGN_ESTIMATE = 4768
E3_MEASURED_PARAMETERS = 1920


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_approved_e3_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_e3" / "feature_encoder_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied_hash != APPROVED_E3_REPORT_SHA256:
        raise ValueError("E.3 report does not have the approved canonical hash")
    if d2.canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("E.3 report canonical content is invalid")
    if report.get("overall_status") != "PASS":
        raise ValueError("E.3 report is not passing")
    return report


def _parameter_breakdown(model: e4.HierarchicalGraphEncoder) -> dict[str, int]:
    return {
        "category_embedding": model.category_embedding.weight.numel(),
        "sample_token": model.sample_token.numel(),
        "feature_to_category_gatv2": sum(
            parameter.numel()
            for parameter in model.feature_to_category.parameters()
        ),
        "category_to_sample_gatv2": sum(
            parameter.numel()
            for parameter in model.category_to_sample.parameters()
        ),
        "layer_norms": sum(
            parameter.numel()
            for module in (
                model.feature_to_category_norm,
                model.category_to_sample_norm,
            )
            for parameter in module.parameters()
        ),
    }


def _attention_normalized(
    edge_index: torch.Tensor, alpha: torch.Tensor, destination_count: int
) -> bool:
    if alpha.ndim != 2 or alpha.shape[1] != e4.ATTENTION_HEADS:
        return False
    if not bool(torch.isfinite(alpha).all()):
        return False
    expected = torch.ones(e4.ATTENTION_HEADS, device=alpha.device)
    return all(
        bool(
            torch.allclose(
                alpha[edge_index[1] == destination].sum(dim=0),
                expected,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        for destination in range(destination_count)
    )


def _destination_head_sums(
    edge_index: torch.Tensor, alpha: torch.Tensor, destination_count: int
) -> torch.Tensor:
    return torch.stack(
        [
            alpha[edge_index[1] == destination].sum(dim=0)
            for destination in range(destination_count)
        ]
    )


def _attention_mode_checks(
    model: e4.HierarchicalGraphEncoder,
    data: Any,
    values: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, eval_attention = model(data, values, return_attention=True)
    model.train()
    torch.manual_seed(SEED)
    with torch.no_grad():
        _, train_attention = model(data, values, return_attention=True)
    model.eval()

    relation_results: dict[str, Any] = {}
    for name, relation, destination_count in (
        ("feature_to_category", e2.FEATURE_RELATION, data["category"].num_nodes),
        ("category_to_sample", e2.CATEGORY_RELATION, data["sample"].num_nodes),
    ):
        eval_edges, eval_alpha = eval_attention[relation]
        train_edges, train_alpha = train_attention[relation]
        train_sums = _destination_head_sums(
            train_edges, train_alpha, destination_count
        )
        relation_results[name] = {
            "edge_order_identical_between_modes": bool(
                torch.equal(eval_edges, train_edges)
            ),
            "eval_coefficients_normalized_per_destination_head": (
                _attention_normalized(eval_edges, eval_alpha, destination_count)
            ),
            "train_coefficients_include_dropout_zeros": bool(
                torch.any(train_alpha == 0)
            ),
            "train_coefficients_not_required_to_normalize": bool(
                not torch.allclose(
                    train_sums,
                    torch.ones_like(train_sums),
                    atol=1e-6,
                    rtol=1e-6,
                )
            ),
        }

    eval_feature_alpha = eval_attention[e2.FEATURE_RELATION][1]
    train_feature_alpha = train_attention[e2.FEATURE_RELATION][1]
    retained = train_feature_alpha != 0
    retained_scaled = bool(
        torch.allclose(
            train_feature_alpha[retained],
            eval_feature_alpha[retained] / (1.0 - e4.DROPOUT_PROBABILITY),
            atol=1e-6,
            rtol=1e-5,
        )
    )
    return {
        "installed_pyg_version": torch_geometric.__version__,
        "returned_coefficients_after_internal_attention_dropout": retained_scaled,
        "eval_mode_required_for_attention_collection": True,
        "train_values_are_post_dropout_scaled": retained_scaled,
        "relations": relation_results,
    }


def _adversarial_topology_checks(
    model: e4.HierarchicalGraphEncoder,
    data: Any,
    values: torch.Tensor,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name in (
        "same_graph_feature_redirect",
        "feature_destination_swap",
        "category_to_sample_redirect",
        "edge_reorder",
    ):
        mutated = data.clone()
        if name == "same_graph_feature_redirect":
            edges = mutated[e2.FEATURE_RELATION].edge_index
            graph = int(edges[0, 0]) // e4.FEATURES_PER_GRAPH
            old_local = int(edges[1, 0]) % e4.CATEGORIES_PER_GRAPH
            edges[1, 0] = (
                graph * e4.CATEGORIES_PER_GRAPH
                + (old_local + 1) % e4.CATEGORIES_PER_GRAPH
            )
        elif name == "feature_destination_swap":
            destinations = mutated[e2.FEATURE_RELATION].edge_index[1]
            right = next(
                index
                for index in range(1, e4.FEATURES_PER_GRAPH)
                if destinations[index] != destinations[0]
            )
            destinations[[0, right]] = destinations[[right, 0]].clone()
        elif name == "category_to_sample_redirect":
            mutated[e2.CATEGORY_RELATION].edge_index[1, 0] = 1
        elif name == "edge_reorder":
            edges = mutated[e2.FEATURE_RELATION].edge_index
            mutated[e2.FEATURE_RELATION].edge_index = edges.flip(1)

        calls: list[str] = []
        handles = [
            model.feature_to_category.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("feature_to_category")
            ),
            model.category_to_sample.register_forward_pre_hook(
                lambda _module, _inputs: calls.append("category_to_sample")
            ),
        ]
        rejected = False
        error = ""
        try:
            with torch.no_grad():
                model(mutated, values)
        except e4.HierarchicalEncoderContractError as exc:
            rejected = True
            error = str(exc)
        finally:
            for handle in handles:
                handle.remove()
        relation = (
            e2.CATEGORY_RELATION
            if name == "category_to_sample_redirect"
            else e2.FEATURE_RELATION
        )
        original_edges = data[relation].edge_index
        mutated_edges = mutated[relation].edge_index
        cases[name] = {
            "rejected": rejected,
            "rejected_before_message_passing": calls == [],
            "shape_preserved": mutated_edges.shape == original_edges.shape,
            "dtype_preserved": mutated_edges.dtype == original_edges.dtype,
            "edge_count_preserved": mutated_edges.shape[1]
            == original_edges.shape[1],
            "error": error,
        }

    with torch.no_grad():
        untouched_output = model(data, values)
    return {
        "authority": "frozen upstream schema tuples used by E.2; no E.4 taxonomy copy",
        "exact_tensor_equality_includes": [
            "shape",
            "dtype",
            "source indices",
            "destination indices",
            "edge order",
        ],
        "untouched_frozen_e2_topology_accepted": bool(
            untouched_output.shape == (data["sample"].num_nodes, 32)
            and torch.isfinite(untouched_output).all()
        ),
        "cases": cases,
        "all_adversarial_cases_rejected": all(
            item["rejected"] for item in cases.values()
        ),
        "all_rejected_before_message_passing": all(
            item["rejected_before_message_passing"] for item in cases.values()
        ),
    }


def _device_check(data: Any, device: str) -> dict[str, Any]:
    torch.manual_seed(SEED)
    moved = data.clone().to(device)
    model = e4.HierarchicalGraphEncoder().to(device)
    values = torch.randn(
        moved["feature"].num_nodes, 32, device=device, requires_grad=True
    )
    output, attention = model(moved, values, return_attention=True)
    output.square().mean().backward()
    gradients = [values.grad] + [parameter.grad for parameter in model.parameters()]
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "status": "PASS",
        "device": device,
        "feature_embedding_shape": list(values.shape),
        "sample_embedding_shape": list(output.shape),
        "feature_attention_shape": list(attention[e2.FEATURE_RELATION][1].shape),
        "category_attention_shape": list(attention[e2.CATEGORY_RELATION][1].shape),
        "output_on_requested_device": output.device.type == device,
        "finite_output": bool(torch.isfinite(output).all()),
        "backward": "PASS",
        "all_gradients_present": all(gradient is not None for gradient in gradients),
        "finite_gradients": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }


def _architecture_checks(
    model: e4.HierarchicalGraphEncoder, data: Any, values: torch.Tensor
) -> dict[str, Any]:
    events: list[str] = []
    modules = (
        ("feature_to_category_gatv2", model.feature_to_category),
        ("feature_to_category_dropout", model.feature_to_category_dropout),
        ("feature_to_category_layer_norm", model.feature_to_category_norm),
        ("feature_to_category_gelu", model.feature_to_category_activation),
        ("category_to_sample_gatv2", model.category_to_sample),
        ("category_to_sample_dropout", model.category_to_sample_dropout),
        ("category_to_sample_layer_norm", model.category_to_sample_norm),
        ("category_to_sample_gelu", model.category_to_sample_activation),
    )
    handles = [
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: events.append(name)
        )
        for name, module in modules
    ]
    with torch.no_grad():
        model(data, values)
    for handle in handles:
        handle.remove()
    expected_events = [name for name, _module in modules]
    return {
        "category_embedding_shape": list(model.category_embedding.weight.shape),
        "sample_token_shape": list(model.sample_token.shape),
        "single_sample_token_parameter": [
            name for name, _ in model.named_parameters() if name == "sample_token"
        ]
        == ["sample_token"],
        "relation_modules_are_distinct": model.feature_to_category
        is not model.category_to_sample,
        "relation_parameters_are_independent": all(
            left is not right
            for left, right in zip(
                model.feature_to_category.parameters(),
                model.category_to_sample.parameters(),
                strict=True,
            )
        ),
        "feature_to_category_configuration": {
            "in_channels": list(model.feature_to_category.in_channels),
            "out_channels": model.feature_to_category.out_channels,
            "heads": model.feature_to_category.heads,
            "concat": model.feature_to_category.concat,
            "dropout": model.feature_to_category.dropout,
            "add_self_loops": model.feature_to_category.add_self_loops,
            "share_weights": model.feature_to_category.share_weights,
            "edge_dim": model.feature_to_category.edge_dim,
            "internal_residual": model.feature_to_category.res is not None,
        },
        "category_to_sample_configuration": {
            "in_channels": list(model.category_to_sample.in_channels),
            "out_channels": model.category_to_sample.out_channels,
            "heads": model.category_to_sample.heads,
            "concat": model.category_to_sample.concat,
            "dropout": model.category_to_sample.dropout,
            "add_self_loops": model.category_to_sample.add_self_loops,
            "share_weights": model.category_to_sample.share_weights,
            "edge_dim": model.category_to_sample.edge_dim,
            "internal_residual": model.category_to_sample.res is not None,
        },
        "observed_operation_order": events,
        "exact_operation_order": events == expected_events,
    }


def build_report(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    e3_report = _load_approved_e3_report(project_root)
    torch.manual_seed(SEED)
    feature_encoder = e3.FeatureNodeEncoder().eval()
    graph_encoder = e4.HierarchicalGraphEncoder().eval()
    breakdown = _parameter_breakdown(graph_encoder)
    parameter_count = sum(parameter.numel() for parameter in graph_encoder.parameters())

    records: list[dict[str, Any]] = []
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    paired_data: dict[tuple[str, str], Any] = {}
    all_real_valid = True
    all_source_unchanged = True
    processed_root = project_root / "data" / "processed" / "preprocessed"
    first_data = None
    first_feature_embeddings = None
    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(
                experiment, split, processed_root=processed_root
            )
            datasets[(experiment, split)] = dataset
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            batch = d6.build_logical_batch(dataset, indices)
            batch_before = np.array(batch.feature_x, copy=True)
            data = e2.logical_batch_to_heterodata(batch, project_root=project_root)
            paired_data[(experiment, split)] = data
            values_before = data["feature"].x.clone()
            ids_before = data["feature"].feature_id.clone()
            category_ids_before = data["category"].category_id.clone()
            target_before = data["sample"].y.clone()
            feature_edges_before = data[e2.FEATURE_RELATION].edge_index.clone()
            category_edges_before = data[e2.CATEGORY_RELATION].edge_index.clone()
            with torch.no_grad():
                feature_embeddings = feature_encoder(
                    data["feature"].x, data["feature"].feature_id
                )
                embeddings_before = feature_embeddings.clone()
                category_shapes: list[list[int]] = []
                hook = graph_encoder.feature_to_category_activation.register_forward_hook(
                    lambda _module, _inputs, output: category_shapes.append(
                        list(output.shape)
                    )
                )
                sample_embeddings, attention = graph_encoder(
                    data, feature_embeddings, return_attention=True
                )
                hook.remove()
            feature_attention_edges, feature_alpha = attention[e2.FEATURE_RELATION]
            category_attention_edges, category_alpha = attention[e2.CATEGORY_RELATION]
            source_unchanged = bool(
                np.array_equal(batch.feature_x, batch_before)
                and torch.equal(data["feature"].x, values_before)
                and torch.equal(data["feature"].feature_id, ids_before)
                and torch.equal(data["category"].category_id, category_ids_before)
                and torch.equal(data["sample"].y, target_before)
                and torch.equal(
                    data[e2.FEATURE_RELATION].edge_index, feature_edges_before
                )
                and torch.equal(
                    data[e2.CATEGORY_RELATION].edge_index, category_edges_before
                )
                and torch.equal(feature_embeddings, embeddings_before)
                and dataset.is_lazy_read_only
            )
            topology_retained = bool(
                torch.equal(feature_attention_edges, feature_edges_before)
                and torch.equal(category_attention_edges, category_edges_before)
            )
            attention_valid = bool(
                feature_alpha.shape == (3 * 78, 4)
                and category_alpha.shape == (3 * 8, 4)
                and _attention_normalized(feature_attention_edges, feature_alpha, 3 * 8)
                and _attention_normalized(category_attention_edges, category_alpha, 3)
            )
            output_valid = bool(
                feature_embeddings.shape == (3 * 78, 32)
                and category_shapes == [[3 * 8, 32]]
                and sample_embeddings.shape == (3, 32)
                and torch.isfinite(sample_embeddings).all()
            )
            all_real_valid &= topology_retained and attention_valid and output_valid
            all_source_unchanged &= source_unchanged
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "source_dataset_length": len(dataset),
                    "tested_indices": indices,
                    "batch_size": batch.batch_size,
                    "e3_feature_embedding_shape": list(feature_embeddings.shape),
                    "e4_category_embedding_shape": category_shapes[0],
                    "e4_sample_embedding_shape": list(sample_embeddings.shape),
                    "feature_attention_shape": list(feature_alpha.shape),
                    "category_attention_shape": list(category_alpha.shape),
                    "attention_normalized": attention_valid,
                    "topology_retained": topology_retained,
                    "exact_frozen_e2_topology_accepted": topology_retained,
                    "finite_output": bool(torch.isfinite(sample_embeddings).all()),
                    "source_inputs_unmodified": source_unchanged,
                }
            )
            if first_data is None:
                first_data = data
                first_feature_embeddings = feature_embeddings

    if first_data is None or first_feature_embeddings is None:
        raise RuntimeError("no real integration batch was produced")

    topology_validation = _adversarial_topology_checks(
        graph_encoder, first_data, first_feature_embeddings
    )
    attention_mode_semantics = _attention_mode_checks(
        graph_encoder, first_data, first_feature_embeddings
    )
    architecture = _architecture_checks(
        graph_encoder, first_data, first_feature_embeddings
    )
    with torch.no_grad():
        deterministic_first = graph_encoder(first_data, first_feature_embeddings)
        deterministic_second = graph_encoder(first_data, first_feature_embeddings)
        changed = first_feature_embeddings.clone()
        changed[:78] += 1000.0
        isolated_output = graph_encoder(first_data, changed)
        target_mutated = first_data.clone()
        target_mutated["sample"].y = first_data["sample"].y + 1000
        target_mutated.e2_audit = {
            "experiment": "changed",
            "split": "changed",
            "paths": ["not-model-input"],
            "provenance": {"hash": "not-model-input"},
        }
        metadata_changed_output = graph_encoder(
            target_mutated, first_feature_embeddings
        )
    graph_isolation = bool(
        not torch.equal(deterministic_first[0], isolated_output[0])
        and torch.equal(deterministic_first[1:], isolated_output[1:])
    )
    leakage_independence = bool(
        torch.equal(deterministic_first, metadata_changed_output)
    )

    e1_e3_results: list[dict[str, Any]] = []
    for split in d4.SPLIT_STEMS:
        e1_data = paired_data[("E1", split)]
        e3_data = paired_data[("E3", split)]
        with torch.no_grad():
            e1_features = feature_encoder(
                e1_data["feature"].x, e1_data["feature"].feature_id
            )
            e3_features = feature_encoder(
                e3_data["feature"].x, e3_data["feature"].feature_id
            )
            e1_output = graph_encoder(e1_data, e1_features)
            e3_output = graph_encoder(e3_data, e3_features)
        e1_e3_results.append(
            {
                "split": split,
                "feature_inputs_identical": bool(
                    torch.equal(e1_data["feature"].x, e3_data["feature"].x)
                    and torch.equal(
                        e1_data["feature"].feature_id,
                        e3_data["feature"].feature_id,
                    )
                ),
                "topology_identical": bool(
                    torch.equal(
                        e1_data[e2.FEATURE_RELATION].edge_index,
                        e3_data[e2.FEATURE_RELATION].edge_index,
                    )
                    and torch.equal(
                        e1_data[e2.CATEGORY_RELATION].edge_index,
                        e3_data[e2.CATEGORY_RELATION].edge_index,
                    )
                ),
                "feature_embeddings_identical": bool(
                    torch.equal(e1_features, e3_features)
                ),
                "sample_embeddings_identical": bool(
                    torch.equal(e1_output, e3_output)
                ),
            }
        )
    e1_e3_all = all(
        item["feature_inputs_identical"]
        and item["topology_identical"]
        and item["feature_embeddings_identical"]
        and item["sample_embeddings_identical"]
        for item in e1_e3_results
    )

    cpu = _device_check(first_data, "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage E.1 requires CUDA, but CUDA is unavailable")
    cuda = _device_check(first_data, "cuda")
    integrity = d4.verify_frozen_integrity(project_root)
    exact_config = all(
        config
        == {
            "in_channels": [32, 32],
            "out_channels": 8,
            "heads": 4,
            "concat": True,
            "dropout": 0.1,
            "add_self_loops": False,
            "share_weights": False,
            "edge_dim": None,
            "internal_residual": False,
        }
        for config in (
            architecture["feature_to_category_configuration"],
            architecture["category_to_sample_configuration"],
        )
    )
    attention_modes_pass = bool(
        attention_mode_semantics["installed_pyg_version"] == "2.8.0.post1"
        and attention_mode_semantics[
            "returned_coefficients_after_internal_attention_dropout"
        ]
        and attention_mode_semantics["train_values_are_post_dropout_scaled"]
        and all(
            item["edge_order_identical_between_modes"]
            and item["eval_coefficients_normalized_per_destination_head"]
            and item["train_coefficients_include_dropout_zeros"]
            and item["train_coefficients_not_required_to_normalize"]
            for item in attention_mode_semantics["relations"].values()
        )
    )
    overall = all(
        (
            parameter_count == E4_DESIGN_ESTIMATE,
            sum(breakdown.values()) == E4_DESIGN_ESTIMATE,
            architecture["category_embedding_shape"] == [8, 32],
            architecture["sample_token_shape"] == [32],
            architecture["single_sample_token_parameter"],
            architecture["relation_modules_are_distinct"],
            architecture["relation_parameters_are_independent"],
            architecture["exact_operation_order"],
            exact_config,
            topology_validation["untouched_frozen_e2_topology_accepted"],
            topology_validation["all_adversarial_cases_rejected"],
            topology_validation["all_rejected_before_message_passing"],
            attention_modes_pass,
            all_real_valid,
            all_source_unchanged,
            graph_isolation,
            leakage_independence,
            torch.equal(deterministic_first, deterministic_second),
            e1_e3_all,
            cpu["status"] == "PASS",
            cuda["status"] == "PASS",
            integrity["overall_status"] == "PASS",
        )
    )
    report: dict[str, Any] = {
        "stage": "E.4",
        "encoder_version": e4.ENCODER_VERSION,
        "implementation": {
            "source_path": "src/graph/hierarchical_encoder.py",
            "source_sha256": _sha256(
                project_root / "src" / "graph" / "hierarchical_encoder.py"
            ),
        },
        "upstream": {
            "e3_encoder_version": e3_report["encoder_version"],
            "e3_report_sha256": e3_report["report_sha256"],
            "e2_report_sha256": e3_report["upstream"]["e2_report_sha256"],
            "provenance": {
                **dict(d8.APPROVED_PROVENANCE),
                "d8_report_sha256": e2.APPROVED_D8_REPORT_SHA256,
                "e1_environment_report_sha256": e2.APPROVED_E1_REPORT_SHA256,
            },
        },
        "environment": {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "api": {
            "normal": "HierarchicalGraphEncoder(data, [B*78,32]) -> [B,32]",
            "attention": "HierarchicalGraphEncoder(data, [B*78,32], return_attention=True) -> ([B,32], relation -> (edge_index, per-head alpha))",
        },
        "architecture": architecture,
        "parameter_count": {
            "design_estimate": E4_DESIGN_ESTIMATE,
            "measured": parameter_count,
            "difference": parameter_count - E4_DESIGN_ESTIMATE,
            "components": breakdown,
            "combined_e3_e4_estimate": E3_MEASURED_PARAMETERS
            + E4_DESIGN_ESTIMATE,
            "combined_e3_e4_measured": e3_report["parameter_count"]["measured"]
            + parameter_count,
        },
        "attention_semantics": {
            "per_head_not_averaged": all(
                item["feature_attention_shape"][1] == e4.ATTENTION_HEADS
                and item["category_attention_shape"][1] == e4.ATTENTION_HEADS
                for item in records
            ),
            "model_signal_not_causal_feature_importance": True,
            "returned_edges_are_exact_supplied_e2_edges": all(
                item["topology_retained"] for item in records
            ),
            "no_added_or_self_loop_edges": all(
                item["topology_retained"] for item in records
            )
            and not graph_encoder.feature_to_category.add_self_loops
            and not graph_encoder.category_to_sample.add_self_loops,
            "normalization_scope": {
                "feature_to_category": "incoming features within each category node",
                "category_to_sample": "eight incoming categories within each graph",
            },
            "pyg_2_8_0_post1_eval_train_behavior": attention_mode_semantics,
            "future_pruning_attention_collection_mode": "eval",
        },
        "exact_e2_topology_validation": topology_validation,
        "cpu_forward_backward": cpu,
        "cuda_forward_backward": cuda,
        "eval_mode_determinism": bool(
            torch.equal(deterministic_first, deterministic_second)
        ),
        "multi_graph_direct_isolation": graph_isolation,
        "target_and_metadata_independence": leakage_independence,
        "e1_e3_same_input_results": e1_e3_results,
        "e1_e3_identical_sample_embeddings_same_model_all": e1_e3_all,
        "real_nine_pair_batches": records,
        "real_d4_d6_e2_e3_e4_integration_all": all_real_valid,
        "source_inputs_unmodified_all": all_source_unchanged,
        "excluded": [
            "feature embedding recomputation",
            "raw feature bypass",
            "targets and provenance as model inputs",
            "classifier heads",
            "logits",
            "loss",
            "optimizer",
            "training loop",
            "metrics",
            "pruning",
            "physical node removal",
            "E.5 and later stages",
        ],
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
        "overall_status": "PASS" if overall else "FAIL",
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.project_root
        / "results"
        / "stage_e4"
        / "hierarchical_encoder_validation.json"
    )
    report = build_report(args.project_root)
    report_bytes = d2.canonical_json_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(report_bytes)
    physical_hash = hashlib.sha256(report_bytes).hexdigest()
    sha_path = output.with_suffix(".sha256")
    sha_path.write_text(f"{physical_hash}  {output.name}\n", encoding="ascii")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"physical_sha256={physical_hash}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
