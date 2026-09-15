"""Verify only the isolated Stage E.1 PyTorch/PyG environment.

This script uses synthetic tensors. It never reads Stage C/D data, constructs
the project model, or performs training.
"""

from __future__ import annotations

import json
import platform

import torch
import torch_geometric
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATv2Conv


EXPECTED_TORCH = "2.14.0+cu130"
EXPECTED_PYG = "2.8.0.post1"
EXPECTED_CUDA = "13.0"


def _gatv2_smoke(device: str) -> dict[str, object]:
    torch.manual_seed(20260915)
    source = torch.randn(3, 32, device=device, requires_grad=True)
    target = torch.randn(2, 32, device=device, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 0], [0, 0, 1, 1]], dtype=torch.long, device=device
    )
    module = GATv2Conv(
        in_channels=(32, 32),
        out_channels=8,
        heads=4,
        concat=True,
        add_self_loops=False,
        share_weights=False,
    ).to(device)
    output, (returned_edges, attention) = module(
        (source, target), edge_index, return_attention_weights=True
    )

    assert output.shape == (2, 32)
    assert returned_edges.shape == edge_index.shape
    assert torch.equal(returned_edges, edge_index)
    assert attention.shape == (4, 4)
    assert torch.isfinite(output).all()
    assert torch.isfinite(attention).all()

    output.square().mean().backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    if device == "cuda":
        assert output.is_cuda and attention.is_cuda
        torch.cuda.synchronize()

    return {
        "status": "PASS",
        "output_shape": list(output.shape),
        "attention_shape": list(attention.shape),
        "input_edge_count": edge_index.size(1),
        "returned_edge_count": returned_edges.size(1),
        "edges_unchanged": bool(torch.equal(returned_edges, edge_index)),
        "finite_output": bool(torch.isfinite(output).all()),
        "finite_attention": bool(torch.isfinite(attention).all()),
        "finite_gradients": all(bool(torch.isfinite(g).all()) for g in gradients),
        "parameter_count": sum(p.numel() for p in module.parameters()),
    }


def _heterodata_smoke() -> dict[str, object]:
    graph = HeteroData()
    graph["feature"].x = torch.randn(3, 32)
    graph["category"].x = torch.randn(2, 32)
    graph["sample"].x = torch.randn(1, 32)
    graph["feature", "contributes_to", "category"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 0, 1]], dtype=torch.long
    )
    graph["category", "contributes_to", "sample"].edge_index = torch.tensor(
        [[0, 1], [0, 0]], dtype=torch.long
    )

    assert graph["feature"].num_nodes == 3
    assert graph["category"].num_nodes == 2
    assert graph["sample"].num_nodes == 1
    feature_edges = graph["feature", "contributes_to", "category"].edge_index
    category_edges = graph["category", "contributes_to", "sample"].edge_index
    assert feature_edges.shape == (2, 3)
    assert category_edges.shape == (2, 2)
    assert feature_edges.max(dim=1).values.tolist() == [2, 1]
    assert category_edges.max(dim=1).values.tolist() == [1, 0]

    batch = Batch.from_data_list([graph, graph.clone()])
    assert batch.num_graphs == 2
    assert batch["feature"].x.shape == (6, 32)
    assert batch["category"].x.shape == (4, 32)
    assert batch["sample"].x.shape == (2, 32)
    assert batch["feature", "contributes_to", "category"].edge_index[:, 3:].tolist() == [
        [3, 4, 5],
        [2, 2, 3],
    ]
    assert batch["category", "contributes_to", "sample"].edge_index[:, 2:].tolist() == [
        [2, 3],
        [1, 1],
    ]
    return {
        "status": "PASS",
        "node_types": graph.node_types,
        "edge_types": [list(edge_type) for edge_type in graph.edge_types],
        "batched_graphs": batch.num_graphs,
        "type_local_indexing": "PASS",
    }


def main() -> None:
    assert torch.__version__ == EXPECTED_TORCH
    assert torch_geometric.__version__ == EXPECTED_PYG
    assert torch.version.cuda == EXPECTED_CUDA
    assert torch.cuda.is_available()

    cpu_result = (torch.tensor([1.0, 2.0]) + torch.tensor([3.0, 4.0])).tolist()
    assert cpu_result == [4.0, 6.0]
    cuda_result = (torch.tensor([1.0, 2.0], device="cuda") * 2).tolist()
    assert cuda_result == [2.0, 4.0]
    torch.cuda.synchronize()

    result = {
        "status": "PASS",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "cpu_tensor_operation": cpu_result,
        "cuda_tensor_operation": cuda_result,
        "gatv2_cpu": _gatv2_smoke("cpu"),
        "gatv2_cuda": _gatv2_smoke("cuda"),
        "heterodata": _heterodata_smoke(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
