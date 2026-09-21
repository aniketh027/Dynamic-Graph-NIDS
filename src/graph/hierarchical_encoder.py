"""Stage E.4 hierarchical feature-to-category-to-sample graph encoder.

The encoder consumes frozen Stage E.3 feature embeddings together with the
two frozen Stage E.2 relations.  It does not recompute feature embeddings,
read Stage C or D data, consume targets/provenance, or define prediction and
training components.
"""

from __future__ import annotations

from typing import TypeAlias

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv

from src.graph.pyg_adapter import CATEGORY_RELATION, FEATURE_RELATION
from src.graph import schema as frozen_schema


ENCODER_VERSION = "stage-e4-hierarchical-graph-encoder-v2"
FEATURES_PER_GRAPH = 78
CATEGORIES_PER_GRAPH = 8
EMBEDDING_CHANNELS = 32
ATTENTION_HEADS = 4
HEAD_CHANNELS = 8
DROPOUT_PROBABILITY = 0.10
NODE_TYPES = ("feature", "category", "sample")
EDGE_TYPES = (FEATURE_RELATION, CATEGORY_RELATION)

Relation: TypeAlias = tuple[str, str, str]
RelationAttention: TypeAlias = tuple[torch.Tensor, torch.Tensor]
AttentionByRelation: TypeAlias = dict[Relation, RelationAttention]


class HierarchicalEncoderContractError(ValueError):
    """Raised when E.2 topology or E.3 embeddings violate the E.4 contract."""


def _require_tensor(
    value: object,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise HierarchicalEncoderContractError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise HierarchicalEncoderContractError(
            f"{name} has shape {tuple(value.shape)}; expected {shape}"
        )
    if value.dtype != dtype:
        raise HierarchicalEncoderContractError(
            f"{name} has dtype {value.dtype}; expected {dtype}"
        )
    if value.device != device:
        raise HierarchicalEncoderContractError(
            f"{name} must be on device {device}"
        )
    return value


def _expected_batched_edge_index(
    source_indices: tuple[int, ...],
    destination_indices: tuple[int, ...],
    *,
    batch_size: int,
    source_nodes_per_graph: int,
    destination_nodes_per_graph: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch the authoritative frozen schema relation in graph-major order."""

    base = torch.tensor(
        (source_indices, destination_indices), dtype=torch.long, device=device
    )
    graph_ids = torch.arange(batch_size, dtype=torch.long, device=device)
    offsets = torch.stack(
        (
            graph_ids * source_nodes_per_graph,
            graph_ids * destination_nodes_per_graph,
        )
    ).repeat_interleave(base.shape[1], dim=1)
    return base.repeat(1, batch_size) + offsets


class HierarchicalGraphEncoder(nn.Module):
    """Apply the two approved bipartite GATv2 hierarchy blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(
            CATEGORIES_PER_GRAPH, EMBEDDING_CHANNELS
        )
        self.sample_token = nn.Parameter(torch.empty(EMBEDDING_CHANNELS))
        nn.init.normal_(self.sample_token)

        self.feature_to_category = GATv2Conv(
            in_channels=(EMBEDDING_CHANNELS, EMBEDDING_CHANNELS),
            out_channels=HEAD_CHANNELS,
            heads=ATTENTION_HEADS,
            concat=True,
            dropout=DROPOUT_PROBABILITY,
            add_self_loops=False,
            share_weights=False,
            edge_dim=None,
            residual=False,
        )
        self.category_to_sample = GATv2Conv(
            in_channels=(EMBEDDING_CHANNELS, EMBEDDING_CHANNELS),
            out_channels=HEAD_CHANNELS,
            heads=ATTENTION_HEADS,
            concat=True,
            dropout=DROPOUT_PROBABILITY,
            add_self_loops=False,
            share_weights=False,
            edge_dim=None,
            residual=False,
        )

        self.feature_to_category_dropout = nn.Dropout(DROPOUT_PROBABILITY)
        self.feature_to_category_norm = nn.LayerNorm(EMBEDDING_CHANNELS)
        self.feature_to_category_activation = nn.GELU()
        self.category_to_sample_dropout = nn.Dropout(DROPOUT_PROBABILITY)
        self.category_to_sample_norm = nn.LayerNorm(EMBEDDING_CHANNELS)
        self.category_to_sample_activation = nn.GELU()

    def _validate_inputs(
        self, data: HeteroData, feature_embeddings: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(data, HeteroData):
            raise TypeError("data must be a torch_geometric.data.HeteroData")
        if tuple(data.node_types) != NODE_TYPES:
            raise HierarchicalEncoderContractError(
                f"data node types must be exactly {NODE_TYPES}"
            )
        if tuple(data.edge_types) != EDGE_TYPES:
            raise HierarchicalEncoderContractError(
                f"data edge types must be exactly {EDGE_TYPES}"
            )
        if not isinstance(feature_embeddings, torch.Tensor):
            raise TypeError("feature_embeddings must be a torch.Tensor")
        if feature_embeddings.ndim != 2 or feature_embeddings.shape[1] != EMBEDDING_CHANNELS:
            raise HierarchicalEncoderContractError(
                "feature_embeddings must have shape [B*78,32]"
            )
        if feature_embeddings.shape[0] == 0 or feature_embeddings.shape[0] % FEATURES_PER_GRAPH:
            raise HierarchicalEncoderContractError(
                "feature_embeddings must contain a positive multiple of 78 nodes"
            )
        if feature_embeddings.dtype != torch.float32:
            raise HierarchicalEncoderContractError(
                "feature_embeddings must have dtype torch.float32"
            )

        batch_size = feature_embeddings.shape[0] // FEATURES_PER_GRAPH
        device = feature_embeddings.device
        category_ids = _require_tensor(
            data["category"].category_id,
            "data['category'].category_id",
            shape=(batch_size * CATEGORIES_PER_GRAPH,),
            dtype=torch.long,
            device=device,
        )
        expected_category_ids = torch.arange(
            CATEGORIES_PER_GRAPH, dtype=torch.long, device=device
        ).repeat(batch_size)
        if not torch.equal(category_ids, expected_category_ids):
            raise HierarchicalEncoderContractError(
                "category_id must contain graph-major 0..7 identities"
            )

        if data["feature"].num_nodes != batch_size * FEATURES_PER_GRAPH:
            raise HierarchicalEncoderContractError(
                "feature node count does not match feature_embeddings"
            )
        if data["category"].num_nodes != batch_size * CATEGORIES_PER_GRAPH:
            raise HierarchicalEncoderContractError(
                "category node count does not match B*8"
            )
        if data["sample"].num_nodes != batch_size:
            raise HierarchicalEncoderContractError(
                "sample node count does not match batch size"
            )

        feature_edges = _require_tensor(
            data[FEATURE_RELATION].edge_index,
            "feature-to-category edge_index",
            shape=(2, batch_size * FEATURES_PER_GRAPH),
            dtype=torch.long,
            device=device,
        )
        category_edges = _require_tensor(
            data[CATEGORY_RELATION].edge_index,
            "category-to-sample edge_index",
            shape=(2, batch_size * CATEGORIES_PER_GRAPH),
            dtype=torch.long,
            device=device,
        )
        expected_feature_edges = _expected_batched_edge_index(
            frozen_schema.FEATURE_TO_CATEGORY_SOURCE_INDICES,
            frozen_schema.FEATURE_TO_CATEGORY_DESTINATION_INDICES,
            batch_size=batch_size,
            source_nodes_per_graph=FEATURES_PER_GRAPH,
            destination_nodes_per_graph=CATEGORIES_PER_GRAPH,
            device=device,
        )
        expected_category_edges = _expected_batched_edge_index(
            frozen_schema.CATEGORY_TO_SAMPLE_SOURCE_INDICES,
            frozen_schema.CATEGORY_TO_SAMPLE_DESTINATION_INDICES,
            batch_size=batch_size,
            source_nodes_per_graph=CATEGORIES_PER_GRAPH,
            destination_nodes_per_graph=1,
            device=device,
        )
        if not torch.equal(feature_edges, expected_feature_edges):
            raise HierarchicalEncoderContractError(
                "feature-to-category edge_index does not exactly match the frozen "
                "graph-major E.2 topology"
            )
        if not torch.equal(category_edges, expected_category_edges):
            raise HierarchicalEncoderContractError(
                "category-to-sample edge_index does not exactly match the frozen "
                "graph-major E.2 topology"
            )
        return batch_size, category_ids, feature_edges, category_edges

    def forward(
        self,
        data: HeteroData,
        feature_embeddings: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionByRelation]:
        """Return ``[B,32]`` sample embeddings and optionally per-head attention.

        Attention coefficients are returned exactly as produced by PyG for the
        supplied E.2 edge order.  They are a model signal, not causal feature
        importance.
        """

        if not isinstance(return_attention, bool):
            raise TypeError("return_attention must be a bool")
        batch_size, category_ids, feature_edges, category_edges = (
            self._validate_inputs(data, feature_embeddings)
        )

        category_targets = self.category_embedding(category_ids)
        sample_targets = self.sample_token.unsqueeze(0).expand(batch_size, -1)

        if return_attention:
            category_message, feature_attention = self.feature_to_category(
                (feature_embeddings, category_targets),
                feature_edges,
                return_attention_weights=True,
            )
        else:
            category_message = self.feature_to_category(
                (feature_embeddings, category_targets), feature_edges
            )
        category_message = self.feature_to_category_dropout(category_message)
        category_embeddings = self.feature_to_category_activation(
            self.feature_to_category_norm(category_message + category_targets)
        )

        if return_attention:
            sample_message, category_attention = self.category_to_sample(
                (category_embeddings, sample_targets),
                category_edges,
                return_attention_weights=True,
            )
        else:
            sample_message = self.category_to_sample(
                (category_embeddings, sample_targets), category_edges
            )
        sample_message = self.category_to_sample_dropout(sample_message)
        sample_embeddings = self.category_to_sample_activation(
            self.category_to_sample_norm(sample_message + sample_targets)
        )

        if return_attention:
            return sample_embeddings, {
                FEATURE_RELATION: feature_attention,
                CATEGORY_RELATION: category_attention,
            }
        return sample_embeddings
