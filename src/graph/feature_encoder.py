"""Stage E.3 causal temporal and feature-identity encoders.

The module consumes only the frozen Stage E.2 feature tensors: ordered
five-flow values and categorical feature IDs.  It does not define category or
sample embeddings, graph message passing, prediction heads, or training logic.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ENCODER_VERSION = "stage-e3-causal-feature-encoder-v1"
FEATURE_COUNT = 78
SEQUENCE_LENGTH = 5
TEMPORAL_CHANNELS = 16
FEATURE_ID_CHANNELS = 8
OUTPUT_CHANNELS = 32
LEFT_PADDING = 2
DROPOUT_PROBABILITY = 0.10


class FeatureEncoderContractError(ValueError):
    """Raised when an E.3 tensor violates the frozen feature-node contract."""


def _validate_feature_values(values: torch.Tensor) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError("feature_values must be a torch.Tensor")
    if values.ndim != 2 or values.shape[1] != SEQUENCE_LENGTH:
        raise FeatureEncoderContractError(
            "feature_values must have shape [N,5]"
        )
    if values.shape[0] <= 0:
        raise FeatureEncoderContractError("feature_values must contain at least one node")
    if values.dtype != torch.float32:
        raise FeatureEncoderContractError("feature_values must have dtype torch.float32")


class CausalTemporalEncoder(nn.Module):
    """Shared two-layer causal TCN over five ordered flow values."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=0)
        self.activation = nn.GELU()

    def forward_sequence(self, feature_values: torch.Tensor) -> torch.Tensor:
        """Return all causal temporal positions with shape ``[N,16,5]``."""

        _validate_feature_values(feature_values)
        sequence = feature_values.unsqueeze(1)
        sequence = F.pad(sequence, (LEFT_PADDING, 0))
        sequence = self.activation(self.conv1(sequence))
        sequence = F.pad(sequence, (LEFT_PADDING, 0))
        sequence = self.activation(self.conv2(sequence))
        if sequence.shape != (
            feature_values.shape[0],
            TEMPORAL_CHANNELS,
            SEQUENCE_LENGTH,
        ):
            raise RuntimeError("causal temporal encoder changed the sequence length")
        return sequence

    def forward(self, feature_values: torch.Tensor) -> torch.Tensor:
        """Return the fifth/final temporal representation with shape ``[N,16]``."""

        return self.forward_sequence(feature_values)[:, :, -1]


class FeatureNodeEncoder(nn.Module):
    """Fuse causal temporal state with the categorical identity of each feature."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal_encoder = CausalTemporalEncoder()
        self.feature_embedding = nn.Embedding(FEATURE_COUNT, FEATURE_ID_CHANNELS)
        self.fusion = nn.Linear(
            TEMPORAL_CHANNELS + FEATURE_ID_CHANNELS, OUTPUT_CHANNELS
        )
        self.normalization = nn.LayerNorm(OUTPUT_CHANNELS)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT_PROBABILITY)

    def forward(
        self, feature_values: torch.Tensor, feature_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode E.2 feature nodes from ``[N,5]`` values and ``[N]`` IDs."""

        _validate_feature_values(feature_values)
        if not isinstance(feature_ids, torch.Tensor):
            raise TypeError("feature_ids must be a torch.Tensor")
        if feature_ids.ndim != 1 or feature_ids.shape[0] != feature_values.shape[0]:
            raise FeatureEncoderContractError(
                "feature_ids must have shape [N] matching feature_values"
            )
        if feature_ids.dtype != torch.long:
            raise FeatureEncoderContractError("feature_ids must have dtype torch.long")
        if feature_ids.device != feature_values.device:
            raise FeatureEncoderContractError(
                "feature_ids and feature_values must be on the same device"
            )
        if torch.any(feature_ids < 0) or torch.any(feature_ids >= FEATURE_COUNT):
            raise FeatureEncoderContractError("feature_ids must be in the range [0,77]")

        temporal = self.temporal_encoder(feature_values)
        identity = self.feature_embedding(feature_ids)
        fused = self.fusion(torch.cat((temporal, identity), dim=-1))
        return self.dropout(self.activation(self.normalization(fused)))
