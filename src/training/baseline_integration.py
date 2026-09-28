"""Thin Stage E.9 adapter from frozen E.2 batches to graph-free baselines.

Optimization, losses, metrics, early stopping, and traversal remain owned by
the frozen Stage E.7 harness.  This module only exposes the authoritative
ordered window values to the dense tensor models in :mod:`baselines`.
"""

from __future__ import annotations

from typing import Final, Mapping

import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.training import baselines


FEATURES: Final = 78
TIME_STEPS: Final = 5


class BaselineIntegrationError(ValueError):
    """Raised when a frozen E.2 batch cannot be used by a baseline."""


def _validated_feature_values(
    data: HeteroData, *, task: str
) -> tuple[torch.Tensor, int]:
    if not isinstance(data, HeteroData):
        raise TypeError("baseline training data must be a PyG HeteroData batch")
    audit = getattr(data, "e2_audit", None)
    if not isinstance(audit, Mapping):
        raise BaselineIntegrationError("batch is missing frozen E.2 audit metadata")
    if audit.get("experiment") != task:
        raise BaselineIntegrationError("baseline task does not match batch experiment")
    if audit.get("split") not in {"train", "validation"}:
        raise BaselineIntegrationError("baseline integration rejects test batches")
    indices = audit.get("ordered_source_sample_indices")
    if not isinstance(indices, list) or not indices:
        raise BaselineIntegrationError("batch has no ordered source sample indices")
    batch_size = len(indices)
    values = data["feature"].x
    feature_ids = data["feature"].feature_id
    feature_batch = data["feature"].batch
    expected_ids = torch.arange(FEATURES, device=feature_ids.device).repeat(batch_size)
    expected_batch = torch.arange(
        batch_size, device=feature_batch.device
    ).repeat_interleave(FEATURES)
    if values.dtype != torch.float32 or tuple(values.shape) != (
        batch_size * FEATURES,
        TIME_STEPS,
    ):
        raise BaselineIntegrationError(
            "frozen E.2 feature values must be float32[B*78,5]"
        )
    if feature_ids.dtype != torch.long or not torch.equal(feature_ids, expected_ids):
        raise BaselineIntegrationError("frozen feature-ID order is invalid")
    if feature_batch.dtype != torch.long or not torch.equal(
        feature_batch, expected_batch
    ):
        raise BaselineIntegrationError("frozen graph-major feature ownership is invalid")
    targets = data["sample"].y
    if targets.ndim != 1 or targets.numel() != batch_size:
        raise BaselineIntegrationError("sample targets do not match baseline batch size")
    return values, batch_size


def final_flow_inputs(data: HeteroData, *, task: str) -> torch.Tensor:
    """Return ``[B,78]`` by reading only temporal position five."""

    values, batch_size = _validated_feature_values(data, task=task)
    return values[:, TIME_STEPS - 1].reshape(batch_size, FEATURES)


def ordered_window_inputs(data: HeteroData, *, task: str) -> torch.Tensor:
    """Return the frozen window as temporal-major ``[B,5,78]`` values."""

    values, batch_size = _validated_feature_values(data, task=task)
    return values.reshape(batch_size, FEATURES, TIME_STEPS).transpose(1, 2).contiguous()


class BaselineTrainingAdapter(nn.Module):
    """Make one dense baseline consumable by frozen E.7 epoch helpers.

    The wrapped baseline remains graph-free.  Edge indices, categories,
    attention, topology, labels, and metadata never enter its forward method.
    """

    def __init__(self, model: nn.Module, *, kind: str, task: str) -> None:
        super().__init__()
        if kind not in baselines.BASELINE_KINDS:
            raise ValueError(f"kind must be one of {baselines.BASELINE_KINDS}")
        actual_kind = baselines.model_family(model)
        if actual_kind != kind:
            raise BaselineIntegrationError(
                "declared baseline family does not match the actual model class"
            )
        if getattr(model, "task", None) != task:
            raise BaselineIntegrationError("baseline model/task mismatch")
        self.model = model
        self._kind = actual_kind
        self._task = task

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def task(self) -> str:
        return self._task

    def forward(self, data: HeteroData) -> torch.Tensor:
        inputs = (
            final_flow_inputs(data, task=self.task)
            if self.kind == baselines.FINAL_FLOW
            else ordered_window_inputs(data, task=self.task)
        )
        return self.model(inputs)
