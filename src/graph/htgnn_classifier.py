"""Stage E.5 inference-time HTGNN classifier composition.

The model composes the frozen E.3 feature-node encoder, frozen E.4
hierarchical graph encoder, and one construction-time task head.  It contains
no training, loss, metric, threshold, checkpoint, or pruning behavior.

The construction-time task/output contract is immutable through the supported public model API.
Runtime validation also rejects useful classes of ordinary
or head-only structural corruption as defense in depth.  Arbitrary in-process
Python reflection or monkeypatching -- including rewriting ``__class__``,
``nn.Module._modules``, methods, globals, class dictionaries, or imported
functions -- is outside this model-integrity threat boundary.  A caller with
arbitrary same-process Python execution is not treated as an untrusted model
consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import torch
from torch import nn
from torch_geometric.data import HeteroData

from src.graph.feature_encoder import FeatureNodeEncoder
from src.graph.hierarchical_encoder import (
    AttentionByRelation,
    HierarchicalGraphEncoder,
)


MODEL_VERSION: Final = "stage-e5-htgnn-classifier-v3"
SAMPLE_EMBEDDING_CHANNELS: Final = 32
TASK_OUTPUT_CHANNELS = MappingProxyType({"E1": 15, "E2": 15, "E3": 1})
_PROTECTED_CONTRACT_ATTRIBUTES = frozenset(
    {
        "task",
        "_task",
        "task_id",
        "_task_id",
        "expected_output_width",
        "_expected_output_width",
    }
)


@dataclass(frozen=True)
class _TaskContract:
    task: str
    output_channels: int


class TaskHeadContractError(RuntimeError):
    """Raised when an E.5 instance no longer has its authorized task head."""


def _contract_for(model: "HTGNNClassifier") -> _TaskContract:
    """Resolve task authority from the concrete model type, never instance state."""

    try:
        return _TASK_CONTRACT_BY_TYPE[type(model)]
    except KeyError as exc:
        raise TaskHeadContractError(
            "model type is not an authorized E.5 task-specific type"
        ) from exc


def _validated_head(
    model: "HTGNNClassifier",
) -> tuple[_TaskContract, nn.Linear]:
    """Return the registered head only when its frozen architecture is intact."""

    contract = _contract_for(model)
    registered = model._modules.get("head")
    try:
        resolved = model.head
    except AttributeError as exc:
        raise TaskHeadContractError("classifier head is missing") from exc
    if resolved is not registered:
        raise TaskHeadContractError(
            "classifier head is shadowed or is not the registered module"
        )
    if type(resolved) is not nn.Linear:
        raise TaskHeadContractError("classifier head must be exactly nn.Linear")
    if resolved.in_features != SAMPLE_EMBEDDING_CHANNELS:
        raise TaskHeadContractError("classifier head in_features must be 32")
    if resolved.out_features != contract.output_channels:
        raise TaskHeadContractError(
            f"{contract.task} classifier head out_features must be "
            f"{contract.output_channels}"
        )
    expected_weight_shape = (
        contract.output_channels,
        SAMPLE_EMBEDDING_CHANNELS,
    )
    if not isinstance(resolved.weight, nn.Parameter) or tuple(
        resolved.weight.shape
    ) != expected_weight_shape:
        raise TaskHeadContractError(
            f"classifier head weight must have shape {expected_weight_shape}"
        )
    expected_bias_shape = (contract.output_channels,)
    if not isinstance(resolved.bias, nn.Parameter) or tuple(
        resolved.bias.shape
    ) != expected_bias_shape:
        raise TaskHeadContractError(
            f"classifier head bias must have shape {expected_bias_shape}"
        )
    return contract, resolved


class HTGNNClassifier(nn.Module):
    """Map one frozen E.2 ``HeteroData`` batch to task-specific raw logits.

    ``task`` is fixed when the instance is constructed.  E1 and E2 return
    ``[B,15]`` logits; E3 canonically returns ``[B]`` binary logits.  The
    classifier head receives only the final frozen E.4 sample embedding.
    """

    def __new__(cls, task: str) -> "HTGNNClassifier":
        if cls is HTGNNClassifier:
            if not isinstance(task, str):
                raise TypeError("task must be a string")
            try:
                concrete_type = _TASK_MODEL_TYPES[task]
            except KeyError as exc:
                raise ValueError("task must be exactly one of E1, E2, or E3") from exc
            return super().__new__(concrete_type)
        return super().__new__(cls)

    def __init__(self, task: str) -> None:
        super().__init__()
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        contract = _contract_for(self)
        if task != contract.task:
            raise ValueError(
                f"{type(self).__name__} is bound to task {contract.task}, not {task}"
            )
        self.feature_encoder = FeatureNodeEncoder()
        self.graph_encoder = HierarchicalGraphEncoder()
        self.head = nn.Linear(
            SAMPLE_EMBEDDING_CHANNELS, contract.output_channels
        )

    def __setattr__(self, name: str, value: object) -> None:
        if name in _PROTECTED_CONTRACT_ATTRIBUTES:
            raise AttributeError(
                f"{name} is reserved by the construction-time task contract"
            )
        modules = self.__dict__.get("_modules")
        if name == "head" and modules is not None and "head" in modules:
            raise TaskHeadContractError(
                "classifier head replacement is forbidden after construction"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in _PROTECTED_CONTRACT_ATTRIBUTES or name == "head":
            raise TaskHeadContractError(
                f"{name} cannot be deleted from the construction-time contract"
            )
        super().__delattr__(name)

    @property
    def task(self) -> str:
        """Return the task bound to this concrete model type at construction."""

        return _contract_for(self).task

    def _encode_sample_embeddings(
        self,
        data: HeteroData,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionByRelation]:
        """Run the complete E.3 -> E.4 hierarchy for diagnostics and forward."""

        if not isinstance(data, HeteroData):
            raise TypeError("data must be a torch_geometric.data.HeteroData")
        if not isinstance(return_attention, bool):
            raise TypeError("return_attention must be a bool")
        feature_embeddings = self.feature_encoder(
            data["feature"].x, data["feature"].feature_id
        )
        return self.graph_encoder(
            data,
            feature_embeddings,
            return_attention=return_attention,
        )

    def forward(
        self,
        data: HeteroData,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionByRelation]:
        """Return raw task logits, optionally with the exact E.4 attention payload."""

        contract, head = _validated_head(self)
        encoded = self._encode_sample_embeddings(
            data, return_attention=return_attention
        )
        if return_attention:
            sample_embeddings, attention = encoded
        else:
            sample_embeddings = encoded
        logits = head(sample_embeddings)
        expected_shape = (sample_embeddings.shape[0], contract.output_channels)
        if tuple(logits.shape) != expected_shape:
            raise TaskHeadContractError(
                f"classifier head returned shape {tuple(logits.shape)}; "
                f"expected {expected_shape}"
            )
        if contract.output_channels == 1:
            logits = logits.squeeze(-1)
        if return_attention:
            return logits, attention
        return logits


class _E1HTGNNClassifier(HTGNNClassifier):
    pass


class _E2HTGNNClassifier(HTGNNClassifier):
    pass


class _E3HTGNNClassifier(HTGNNClassifier):
    pass


_TASK_MODEL_TYPES = MappingProxyType(
    {
        "E1": _E1HTGNNClassifier,
        "E2": _E2HTGNNClassifier,
        "E3": _E3HTGNNClassifier,
    }
)
_TASK_CONTRACT_BY_TYPE = MappingProxyType(
    {
        _E1HTGNNClassifier: _TaskContract("E1", 15),
        _E2HTGNNClassifier: _TaskContract("E2", 15),
        _E3HTGNNClassifier: _TaskContract("E3", 1),
    }
)
