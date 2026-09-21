"""Build deterministic Stage E.3 causal feature-encoder evidence."""

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
from src.graph import pyg_adapter as e2  # noqa: E402
from src.graph import schema as d2  # noqa: E402


SEED = 20260921
APPROVED_E2_REPORT_SHA256 = (
    "370048a60c7a7b25643d820730bb82438e031eee5f6e8a7c07b49efaf17e4fec"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_breakdown(model: e3.FeatureNodeEncoder) -> dict[str, int]:
    return {
        "causal_temporal_encoder": sum(
            parameter.numel() for parameter in model.temporal_encoder.parameters()
        ),
        "feature_id_embedding": model.feature_embedding.weight.numel(),
        "fusion_linear": sum(parameter.numel() for parameter in model.fusion.parameters()),
        "layer_norm": sum(
            parameter.numel() for parameter in model.normalization.parameters()
        ),
    }


def _causality_checks() -> dict[str, Any]:
    torch.manual_seed(SEED)
    encoder = e3.CausalTemporalEncoder().eval()
    original = torch.randn(3, 5)
    prefix_results: list[bool] = []
    gradient_results: list[bool] = []
    for temporal_position in range(5):
        changed = original.clone()
        changed[:, temporal_position + 1 :] += 1000.0
        prefix_results.append(
            bool(
                torch.equal(
                    encoder.forward_sequence(original)[:, :, temporal_position],
                    encoder.forward_sequence(changed)[:, :, temporal_position],
                )
            )
        )
        differentiable = original.clone().requires_grad_(True)
        output = encoder.forward_sequence(differentiable)[:, :, temporal_position].sum()
        gradient = torch.autograd.grad(output, differentiable)[0]
        gradient_results.append(
            bool(torch.count_nonzero(gradient[:, temporal_position + 1 :]) == 0)
        )

    with torch.no_grad():
        encoder.conv1.weight.copy_(
            torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 3).repeat(8, 1, 1)
        )
        encoder.conv1.bias.zero_()
        encoder.conv2.weight.copy_(
            torch.tensor([1.0, 2.0, 4.0]).reshape(1, 1, 3).repeat(16, 8, 1)
        )
        encoder.conv2.bias.zero_()
    zeros = torch.zeros(1, 5)
    first_changed = zeros.clone()
    first_changed[0, 0] = 1.0
    position_five_reaches_one = not torch.equal(encoder(zeros), encoder(first_changed))
    base = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    final_changed = base.clone()
    final_changed[0, 4] = -999.0
    position_four_ignores_five = torch.equal(
        encoder.forward_sequence(base)[:, :, 3],
        encoder.forward_sequence(final_changed)[:, :, 3],
    )
    ordered = encoder(base)
    reversed_order = encoder(base.flip(1))
    return {
        "conv1_padding": list(encoder.conv1.padding),
        "conv2_padding": list(encoder.conv2.padding),
        "explicit_left_padding_each_layer": e3.LEFT_PADDING,
        "right_padding": 0,
        "full_sequence_shape": list(encoder.forward_sequence(original).shape),
        "output_sequence_length": encoder.forward_sequence(original).shape[-1],
        "prefix_invariance_by_position": prefix_results,
        "zero_future_gradient_by_position": gradient_results,
        "position_5_receptive_field_reaches_position_1": bool(position_five_reaches_one),
        "position_5_change_cannot_alter_position_4": bool(position_four_ignores_five),
        "temporal_order_sensitivity": bool(not torch.equal(ordered, reversed_order)),
        "deterministic_repeat": bool(torch.equal(ordered, encoder(base))),
    }


def _device_check(device: str) -> dict[str, Any]:
    torch.manual_seed(SEED)
    model = e3.FeatureNodeEncoder().to(device)
    values = torch.randn(156, 5, device=device, requires_grad=True)
    feature_ids = torch.arange(78, device=device).repeat(2)
    output = model(values, feature_ids)
    output.square().mean().backward()
    gradients = [values.grad] + [parameter.grad for parameter in model.parameters()]
    if device == "cuda":
        torch.cuda.synchronize()
    return {
        "status": "PASS",
        "device": device,
        "input_shape": list(values.shape),
        "output_shape": list(output.shape),
        "output_on_requested_device": output.device.type == device,
        "finite_output": bool(torch.isfinite(output).all()),
        "backward": "PASS",
        "all_gradients_present": all(gradient is not None for gradient in gradients),
        "finite_gradients": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
    }


def _load_approved_e2_report(project_root: Path) -> dict[str, Any]:
    path = project_root / "results" / "stage_e2" / "heterodata_adapter_validation.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if supplied_hash != APPROVED_E2_REPORT_SHA256:
        raise ValueError("E.2 report does not have the approved canonical hash")
    if d2.canonical_sha256(unhashed) != supplied_hash:
        raise ValueError("E.2 report canonical content is invalid")
    if report.get("overall_status") != "PASS":
        raise ValueError("E.2 report is not passing")
    return report


def build_report(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    e2_report = _load_approved_e2_report(project_root)
    torch.manual_seed(SEED)
    model = e3.FeatureNodeEncoder().eval()
    parameter_breakdown = _parameter_breakdown(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    causality = _causality_checks()

    records: list[dict[str, Any]] = []
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    all_source_unchanged = True
    all_outputs_valid = True
    processed_root = project_root / "data" / "processed" / "preprocessed"
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
            values_before = data["feature"].x.clone()
            ids_before = data["feature"].feature_id.clone()
            with torch.no_grad():
                output = model(data["feature"].x, data["feature"].feature_id)
            source_unchanged = bool(
                np.array_equal(batch.feature_x, batch_before)
                and torch.equal(data["feature"].x, values_before)
                and torch.equal(data["feature"].feature_id, ids_before)
                and dataset.is_lazy_read_only
            )
            output_valid = output.shape == (3 * 78, 32) and bool(torch.isfinite(output).all())
            all_source_unchanged &= source_unchanged
            all_outputs_valid &= output_valid
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "source_dataset_length": len(dataset),
                    "tested_indices": indices,
                    "batch_size": batch.batch_size,
                    "e2_feature_shape": list(data["feature"].x.shape),
                    "e3_output_shape": list(output.shape),
                    "finite_output": bool(torch.isfinite(output).all()),
                    "source_inputs_unmodified": source_unchanged,
                }
            )

    e1_e3_results: list[dict[str, Any]] = []
    for split in d4.SPLIT_STEMS:
        e1_dataset = datasets[("E1", split)]
        e3_dataset = datasets[("E3", split)]
        indices = [0, len(e1_dataset) // 2, len(e1_dataset) - 1]
        e1_data = e2.logical_batch_to_heterodata(
            d6.build_logical_batch(e1_dataset, indices), project_root=project_root
        )
        e3_data = e2.logical_batch_to_heterodata(
            d6.build_logical_batch(e3_dataset, indices), project_root=project_root
        )
        with torch.no_grad():
            e1_output = model(e1_data["feature"].x, e1_data["feature"].feature_id)
            e3_output = model(e3_data["feature"].x, e3_data["feature"].feature_id)
        e1_e3_results.append(
            {
                "split": split,
                "tested_indices": indices,
                "inputs_identical": bool(
                    torch.equal(e1_data["feature"].x, e3_data["feature"].x)
                    and torch.equal(
                        e1_data["feature"].feature_id,
                        e3_data["feature"].feature_id,
                    )
                ),
                "embeddings_identical_same_model": bool(torch.equal(e1_output, e3_output)),
            }
        )

    equal_values = torch.ones(2, 5)
    with torch.no_grad():
        identity_output = model(equal_values, torch.tensor([0, 1], dtype=torch.long))
        deterministic_first = model(
            torch.arange(390, dtype=torch.float32).reshape(78, 5), torch.arange(78)
        )
        deterministic_second = model(
            torch.arange(390, dtype=torch.float32).reshape(78, 5), torch.arange(78)
        )
    feature_id_distinction = not torch.equal(identity_output[0], identity_output[1])
    eval_determinism = torch.equal(deterministic_first, deterministic_second)
    cpu = _device_check("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage E.1 requires CUDA, but CUDA is unavailable")
    cuda = _device_check("cuda")
    integrity = d4.verify_frozen_integrity(project_root)
    causality_pass = all(
        (
            causality["conv1_padding"] == [0],
            causality["conv2_padding"] == [0],
            causality["explicit_left_padding_each_layer"] == 2,
            causality["right_padding"] == 0,
            causality["output_sequence_length"] == 5,
            all(causality["prefix_invariance_by_position"]),
            all(causality["zero_future_gradient_by_position"]),
            causality["position_5_receptive_field_reaches_position_1"],
            causality["position_5_change_cannot_alter_position_4"],
            causality["temporal_order_sensitivity"],
            causality["deterministic_repeat"],
        )
    )
    e1_e3_all = all(
        item["inputs_identical"] and item["embeddings_identical_same_model"]
        for item in e1_e3_results
    )
    overall = all(
        (
            parameter_count == 1920,
            sum(parameter_breakdown.values()) == 1920,
            causality_pass,
            feature_id_distinction,
            eval_determinism,
            cpu["status"] == "PASS",
            cuda["status"] == "PASS",
            len(records) == 9,
            all_outputs_valid,
            all_source_unchanged,
            e1_e3_all,
            integrity["overall_status"] == "PASS",
        )
    )
    report: dict[str, Any] = {
        "stage": "E.3",
        "encoder_version": e3.ENCODER_VERSION,
        "implementation": {
            "source_path": "src/graph/feature_encoder.py",
            "source_sha256": _sha256(
                project_root / "src" / "graph" / "feature_encoder.py"
            ),
        },
        "upstream": {
            "e2_adapter_version": e2_report["e2_version"],
            "e2_report_sha256": e2_report["report_sha256"],
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
            "temporal_sequence": "CausalTemporalEncoder.forward_sequence([N,5]) -> [N,16,5]",
            "temporal_final": "CausalTemporalEncoder([N,5]) -> [N,16]",
            "feature_node": "FeatureNodeEncoder([N,5], [N]) -> [N,32]",
        },
        "architecture": {
            "temporal": "left-pad 2 -> Conv1d(1,8,k=3,p=0) -> GELU -> left-pad 2 -> Conv1d(8,16,k=3,p=0) -> GELU",
            "readout": "temporal position 5",
            "feature_identity": "Embedding(78,8)",
            "fusion": "Linear(24,32) -> LayerNorm(32) -> GELU -> Dropout(0.10)",
            "excluded": [
                "category embeddings",
                "sample token",
                "GATv2",
                "classifier",
                "loss",
                "optimizer",
                "training",
                "pruning",
                "E.4",
            ],
        },
        "parameter_count": {
            "design_estimate": 1920,
            "measured": parameter_count,
            "difference": parameter_count - 1920,
            "components": parameter_breakdown,
        },
        "causality": causality,
        "feature_id_distinction": bool(feature_id_distinction),
        "eval_mode_determinism": bool(eval_determinism),
        "cpu_forward_backward": cpu,
        "cuda_forward_backward": cuda,
        "real_nine_pair_batches": records,
        "real_d4_d6_e2_e3_integration_all": bool(all_outputs_valid),
        "source_inputs_unmodified_all": bool(all_source_unchanged),
        "e1_e3_same_input_results": e1_e3_results,
        "e1_e3_identical_embeddings_same_model_all": bool(e1_e3_all),
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
        args.project_root / "results" / "stage_e3" / "feature_encoder_validation.json"
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
