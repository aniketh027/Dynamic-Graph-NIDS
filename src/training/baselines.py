"""Stage E.9 graph-free MLP baseline models.

The models in this module accept only dense tensors.  They deliberately know
nothing about graph topology, PyG containers, preprocessing, split membership,
labels, losses, optimizers, checkpoints, metrics, attention, or pruning.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

import torch
from torch import nn


BASELINE_VERSION: Final = "stage-e9-frozen-mlp-baselines-v1"
TASKS: Final = ("E1", "E2", "E3")
FINAL_FLOW: Final = "final_flow_mlp"
FLATTENED_FIVE_FLOW: Final = "flattened_five_flow_mlp"
BASELINE_KINDS: Final = (FINAL_FLOW, FLATTENED_FIVE_FLOW)
TASK_OUTPUT_CHANNELS = MappingProxyType({"E1": 15, "E2": 15, "E3": 1})
EXPECTED_PARAMETER_COUNTS = MappingProxyType(
    {
        FINAL_FLOW: MappingProxyType({"E1": 7_823, "E2": 7_823, "E3": 7_361}),
        FLATTENED_FIVE_FLOW: MappingProxyType(
            {"E1": 27_791, "E2": 27_791, "E3": 27_329}
        ),
    }
)


class BaselineContractError(ValueError):
    """Raised when an input or constructed baseline violates Stage E.9."""


def _validate_task(task: object) -> str:
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if task not in TASKS:
        raise ValueError("task must be exactly one of E1, E2, or E3")
    return task


class _FrozenMLP(nn.Module):
    """Exact shared hidden architecture for the two approved baselines."""

    input_width: int
    input_shape_tail: tuple[int, ...]
    baseline_kind: str

    def __init__(
        self,
        task: str,
        *,
        input_width: int,
        input_shape_tail: tuple[int, ...],
        baseline_kind: str,
    ) -> None:
        super().__init__()
        self._task = _validate_task(task)
        self._input_width = input_width
        self._input_shape_tail = input_shape_tail
        self._baseline_kind = baseline_kind
        self.hidden = nn.Sequential(
            nn.Linear(input_width, 64, bias=True),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32, bias=True),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.head = nn.Linear(32, TASK_OUTPUT_CHANNELS[self._task], bias=True)
        measured = parameter_count(self)
        expected = EXPECTED_PARAMETER_COUNTS[baseline_kind][self._task]
        if measured != expected:
            raise BaselineContractError(
                f"{baseline_kind}/{self._task} parameter count changed: "
                f"{measured} != {expected}"
            )

    @property
    def task(self) -> str:
        return self._task

    @property
    def input_width(self) -> int:
        return self._input_width

    @property
    def input_shape_tail(self) -> tuple[int, ...]:
        return self._input_shape_tail

    @property
    def baseline_kind(self) -> str:
        return self._baseline_kind

    def _validate_input(self, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError("baseline input must be a torch.Tensor")
        if values.dtype != torch.float32:
            raise BaselineContractError("baseline input must have dtype torch.float32")
        if values.ndim != 1 + len(self.input_shape_tail) or tuple(
            values.shape[1:]
        ) != self.input_shape_tail:
            expected = "[B," + ",".join(str(v) for v in self.input_shape_tail) + "]"
            raise BaselineContractError(
                f"{self.baseline_kind} input must have shape {expected}; "
                f"got {tuple(values.shape)}"
            )
        if values.shape[0] <= 0:
            raise BaselineContractError("empty baseline batches are not supported")

    def _flatten(self, values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return task-specific raw logits without a probability activation."""

        self._validate_input(values)
        flattened = self._flatten(values)
        if tuple(flattened.shape) != (values.shape[0], self.input_width):
            raise BaselineContractError("baseline flattening violated its input width")
        logits = self.head(self.hidden(flattened))
        expected = (values.shape[0], TASK_OUTPUT_CHANNELS[self._task])
        if tuple(logits.shape) != expected:
            raise BaselineContractError(
                f"baseline head returned {tuple(logits.shape)}; expected {expected}"
            )
        return logits.squeeze(-1) if self._task == "E3" else logits


class FinalFlowMLP(_FrozenMLP):
    """Consume exactly the fifth/final flow as a ``float32[B,78]`` tensor."""

    def __init__(self, task: str) -> None:
        super().__init__(
            task,
            input_width=78,
            input_shape_tail=(78,),
            baseline_kind=FINAL_FLOW,
        )

    def _flatten(self, values: torch.Tensor) -> torch.Tensor:
        return values


class FlattenedFiveFlowMLP(_FrozenMLP):
    """Flatten an ordered ``float32[B,5,78]`` window in temporal-major order."""

    def __init__(self, task: str) -> None:
        super().__init__(
            task,
            input_width=390,
            input_shape_tail=(5, 78),
            baseline_kind=FLATTENED_FIVE_FLOW,
        )

    def _flatten(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(values.shape[0], 390)


def create_baseline(kind: str, task: str) -> _FrozenMLP:
    """Construct one exact task-specific Stage E.9 baseline."""

    if kind == FINAL_FLOW:
        return FinalFlowMLP(task)
    if kind == FLATTENED_FIVE_FLOW:
        return FlattenedFiveFlowMLP(task)
    raise ValueError(f"kind must be one of {BASELINE_KINDS}")


def model_family(model: nn.Module) -> str:
    """Derive and verify the immutable family of a supported baseline instance."""

    if type(model) is FinalFlowMLP:
        expected = FINAL_FLOW
    elif type(model) is FlattenedFiveFlowMLP:
        expected = FLATTENED_FIVE_FLOW
    else:
        raise BaselineContractError("unsupported baseline model class")
    if model.baseline_kind != expected:
        raise BaselineContractError("baseline class and family identity disagree")
    return expected


def parameter_count(model: nn.Module) -> int:
    """Return the actual learned-parameter count without architecture changes."""

    return sum(parameter.numel() for parameter in model.parameters())
