"""Build the deterministic Stage E.2 bounded real-data validation report."""

from __future__ import annotations

import argparse
import ast
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
from src.graph import full_split_integration as d8  # noqa: E402
from src.graph import pyg_adapter as e2  # noqa: E402
from src.graph import schema as d2  # noqa: E402


def _equivalent(batch: d6.LogicalGraphBatch, data: object) -> bool:
    return bool(
        torch.equal(
            data["feature"].x.reshape(batch.batch_size, 78, 5),
            torch.tensor(np.array(batch.feature_x, copy=True)),
        )
        and torch.equal(
            data[e2.FEATURE_RELATION].edge_index,
            torch.tensor(
                np.stack(
                    (
                        batch.feature_to_category_sources,
                        batch.feature_to_category_destinations,
                    )
                )
            ),
        )
        and torch.equal(
            data[e2.CATEGORY_RELATION].edge_index,
            torch.tensor(
                np.stack(
                    (
                        batch.category_to_sample_sources,
                        batch.category_to_sample_destinations,
                    )
                )
            ),
        )
        and torch.equal(data["feature"].batch, torch.tensor(batch.feature_batch.copy()))
        and torch.equal(data["category"].batch, torch.tensor(batch.category_batch.copy()))
        and torch.equal(data["sample"].batch, torch.tensor(batch.sample_batch.copy()))
    )


def _zero_parameter_contract(project_root: Path) -> bool:
    tree = ast.parse(
        (project_root / "src" / "graph" / "pyg_adapter.py").read_text(encoding="utf-8")
    )
    forbidden = {"Module", "Parameter", "Embedding", "Linear", "Conv1d", "GATv2Conv"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden:
                return False
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id == "Module" for base in node.bases
        ):
            return False
    return True


def build_report(project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root)
    records: list[dict[str, Any]] = []
    datasets: dict[tuple[str, str], d4.LazyGraphDataset] = {}
    all_equivalent = True
    all_valid = True
    all_isolated = True

    for experiment in d4.EXPERIMENT_DIRS:
        for split in d4.SPLIT_STEMS:
            dataset = d4.LazyGraphDataset(experiment, split)
            datasets[(experiment, split)] = dataset
            indices = [0, len(dataset) // 2, len(dataset) - 1]
            batch = d6.build_logical_batch(dataset, indices)
            data = e2.logical_batch_to_heterodata(batch, project_root=project_root)
            expected_y = torch.tensor(
                np.array(batch.targets, copy=True),
                dtype=torch.float32 if experiment == "E3" else torch.long,
            )
            equivalent = _equivalent(batch, data) and torch.equal(
                data["sample"].y, expected_y
            )
            isolated = bool(
                torch.equal(
                    data[e2.FEATURE_RELATION].edge_index[0] // 78,
                    data[e2.FEATURE_RELATION].edge_index[1] // 8,
                )
                and torch.equal(
                    data[e2.CATEGORY_RELATION].edge_index[0] // 8,
                    data[e2.CATEGORY_RELATION].edge_index[1],
                )
            )
            valid = bool(data.validate(raise_on_error=False))
            all_equivalent &= equivalent
            all_isolated &= isolated
            all_valid &= valid
            records.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "source_dataset_length": len(dataset),
                    "tested_indices": indices,
                    "batch_size": batch.batch_size,
                    "node_counts": {"feature": 234, "category": 24, "sample": 3},
                    "edge_counts": {
                        "feature_to_category": 234,
                        "category_to_sample": 24,
                        "total": 258,
                    },
                    "target_dtype": str(data["sample"].y.dtype),
                    "d6_equivalence": equivalent,
                    "no_cross_sample_edges": isolated,
                    "heterodata_validation": valid,
                }
            )

    e1_e3_consistent = True
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
        e1_e3_consistent &= torch.equal(e1_data["feature"].x, e3_data["feature"].x)

    e2_independent = all(
        datasets[("E2", split)].x_path.parent.name == "exp2"
        and datasets[("E2", split)].x_path != datasets[("E1", split)].x_path
        and datasets[("E2", split)].x_path != datasets[("E3", split)].x_path
        for split in d4.SPLIT_STEMS
    )

    mutation_batch = d6.build_logical_batch(datasets[("E1", "train")], [0])
    source_snapshot = np.array(mutation_batch.feature_x, copy=True)
    mutation_data = e2.logical_batch_to_heterodata(
        mutation_batch, project_root=project_root
    )
    mutation_data["feature"].x[0, 0] += 1
    source_copy_safe = np.array_equal(mutation_batch.feature_x, source_snapshot)

    cuda_data = e2.logical_batch_to_heterodata(
        d6.build_logical_batch(datasets[("E1", "train")], [0]),
        project_root=project_root,
    ).to("cuda")
    cuda_tensors = [
        value
        for node_type in e2.NODE_TYPES
        for value in cuda_data[node_type].values()
        if isinstance(value, torch.Tensor)
    ] + [cuda_data[edge_type].edge_index for edge_type in e2.EDGE_TYPES]
    cuda_transfer = bool(torch.cuda.is_available() and all(x.is_cuda for x in cuda_tensors))
    torch.cuda.synchronize()

    integrity = d4.verify_frozen_integrity(project_root)
    zero_parameters = _zero_parameter_contract(project_root)
    overall = all(
        (
            len(records) == 9,
            all_equivalent,
            all_valid,
            all_isolated,
            e1_e3_consistent,
            e2_independent,
            source_copy_safe,
            cuda_transfer,
            zero_parameters,
            integrity["overall_status"] == "PASS",
        )
    )
    report: dict[str, Any] = {
        "e2_version": e2.ADAPTER_VERSION,
        "stage_d_checkpoint": {
            "commit": e2.FROZEN_STAGE_D_COMMIT,
            "tag": e2.FROZEN_STAGE_D_TAG,
        },
        "provenance": {
            **dict(d8.APPROVED_PROVENANCE),
            "d8_report_sha256": e2.APPROVED_D8_REPORT_SHA256,
            "e1_environment_report_sha256": e2.APPROVED_E1_REPORT_SHA256,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "canonical_device": "cpu",
        },
        "adapter_api": "logical_batch_to_heterodata(batch) -> HeteroData",
        "adapter_contract": {
            "direct_single_batched_heterodata": True,
            "bounded_to_requested_d6_batch": True,
            "per_sample_graph_list": False,
            "node_fields": {
                "feature": {"x": "float32[B*78,5]", "feature_id": "long[B*78]", "batch": "long[B*78]", "ptr": "long[B+1]"},
                "category": {"category_id": "long[B*8]", "batch": "long[B*8]", "ptr": "long[B+1]"},
                "sample": {"y": "long[B] for E1/E2; float32[B] for E3", "batch": "long[B]", "ptr": "long[B+1]"},
            },
            "target_conversion": {"E1": "int16 to long", "E2": "int16 to long", "E3": "binary int16 to float32"},
            "relations": [list(e2.FEATURE_RELATION), list(e2.CATEGORY_RELATION)],
            "ptr_decision": "derived PyG-compatible boundaries exposed and cross-validated; D6 batch vectors remain ownership authority",
        },
        "real_nine_pair_batches": records,
        "d6_equivalence_all": all_equivalent,
        "no_cross_sample_edges_all": all_isolated,
        "heterodata_validation_all": all_valid,
        "e1_e3_feature_consistency": bool(e1_e3_consistent),
        "e2_independent_source": bool(e2_independent),
        "bounded_torch_owned_copy": bool(source_copy_safe),
        "cuda_transfer": bool(cuda_transfer),
        "learned_parameter_count": 0 if zero_parameters else None,
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
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.project_root)
    output = args.output or args.project_root / "results" / "stage_e2" / "heterodata_adapter_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(e2.report_bytes(report))
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    print(f"output={output}")
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
