"""Feature scaling and finite-value replacement for Stage C.

All statistics (RobustScaler medians/IQRs, per-column finite replacements)
are LEARNED from the training split of the experiment to which they apply.
E1 and E3 share the E1 state because their feature splits are identical; E2
uses a separate state fitted on E2 train because its membership differs.

This guarantees that no information from val/test leaks into the scaler.

Sentinel preservation
---------------------
CICFlowMeter emits ``-1`` as a "not observed" sentinel in two specific columns:

* ``Init Fwd Win Byts``
* ``Init Bwd Win Byts``

These ``-1`` values are NOT replaced, clipped, or scaled away.  They carry a
distinct semantic meaning and must survive the transform pipeline unchanged.

NaN / inf policy (applied before scaling)
-----------------------------------------
The Stage C cleaning/windowing path intentionally preserves raw NaN and +/-inf
until after splitting.  They are then replaced using train-derived column-wise
statistics: NaN → train column median, inf → train column max/min finite value.
No file-local statistic and no validation/test value participates in fitting.

Scaling
-------
``sklearn.preprocessing.RobustScaler`` is applied per feature dimension across
all T*N_train windows flattened to (N_train*T, 78).  Each experiment-specific
scaler is fit once on its flattened training windows and applied to that
experiment's validation and test splits.

The scaler state is persisted as a ``scaler_state.json`` artefact (medians,
IQR scales, feature names) so it can be reconstructed without pickling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler

from src.preprocessing.schema import (
    MODEL_FEATURE_COLUMNS,
    SENTINEL_NEGATIVE_COLUMNS,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Indices of the sentinel columns within MODEL_FEATURE_COLUMNS.
_SENTINEL_INDICES: Tuple[int, ...] = tuple(
    MODEL_FEATURE_COLUMNS.index(col) for col in SENTINEL_NEGATIVE_COLUMNS
)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

@dataclass
class FitTransformState:
    """Learned statistics from the training split."""

    #: Feature column names in the order used for the 78-dim axis.
    feature_names: List[str]
    #: RobustScaler center_ (median), shape (78,).
    center: np.ndarray
    #: RobustScaler scale_ (IQR), shape (78,).
    scale: np.ndarray
    #: Per-column train-median used as NaN fallback, shape (78,).
    train_median: np.ndarray
    #: Per-column train max-finite used as +inf fallback, shape (78,).
    train_max_finite: np.ndarray
    #: Per-column train min-finite used as -inf fallback, shape (78,).
    train_min_finite: np.ndarray

    # Internal: the fitted sklearn scaler kept for transform().
    _scaler: Optional[RobustScaler] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable representation."""
        return {
            "feature_names": self.feature_names,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "train_median": self.train_median.tolist(),
            "train_max_finite": self.train_max_finite.tolist(),
            "train_min_finite": self.train_min_finite.tolist(),
            "sentinel_columns": list(SENTINEL_NEGATIVE_COLUMNS),
            "sentinel_policy": "-1 values in sentinel columns are preserved unchanged",
            "scaler": "sklearn RobustScaler (quantile_range=(25.0, 75.0))",
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "FitTransformState":
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        state = cls(
            feature_names=d["feature_names"],
            center=np.array(d["center"], dtype=np.float64),
            scale=np.array(d["scale"], dtype=np.float64),
            train_median=np.array(d["train_median"], dtype=np.float64),
            train_max_finite=np.array(d["train_max_finite"], dtype=np.float64),
            train_min_finite=np.array(d["train_min_finite"], dtype=np.float64),
        )
        # Rebuild the sklearn scaler from the stored statistics.
        scaler = RobustScaler()
        scaler.center_ = state.center.copy()
        scaler.scale_ = state.scale.copy()
        scaler.n_features_in_ = len(state.feature_names)
        state._scaler = scaler
        return state


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_transforms(X_train: np.ndarray) -> FitTransformState:
    """Fit scaling statistics on the training windows.

    Parameters
    ----------
    X_train:
        Float32 array of shape ``(n_train_windows, T, 78)``.  Only finite,
        non-sentinel values contribute to the scaler fit.

    Returns
    -------
    FitTransformState
        All statistics needed to transform any split.
    """
    n_win, T, n_feat = X_train.shape
    assert n_feat == len(MODEL_FEATURE_COLUMNS), (
        f"Expected {len(MODEL_FEATURE_COLUMNS)} features, got {n_feat}"
    )

    # Keep the full tensor as a view/memmap.  Materialising the flattened E1
    # train split as float64 used >6 GiB before RobustScaler made its own work
    # arrays, which is not viable on the target 16-GiB machine.
    flat = X_train.reshape(-1, n_feat)

    return _fit_from_column_loader(
        n_feat,
        lambda j: np.asarray(flat[:, j], dtype=np.float64),
    )


def fit_transforms_indexed(
    X: np.ndarray,
    window_indices: np.ndarray,
    chunk_windows: int = 100_000,
) -> FitTransformState:
    """Fit a transform on an index-selected training split without copying it.

    E2 uses fancy indices into the global window tensor.  Building ``X[idx]``
    for all E2 training windows would allocate roughly 3.4 GiB.  This function
    gathers one feature column at a time in bounded chunks, preserving the
    exact E2-train membership while keeping peak memory bounded.
    """
    indices = np.asarray(window_indices, dtype=np.int64)
    _require_shape = X.ndim == 3 and X.shape[2] == len(MODEL_FEATURE_COLUMNS)
    if not _require_shape:
        raise ValueError(
            f"Expected X shape (N, T, {len(MODEL_FEATURE_COLUMNS)}), got {X.shape}"
        )
    if chunk_windows <= 0:
        raise ValueError("chunk_windows must be positive")

    T = X.shape[1]
    n_values = len(indices) * T

    def _load_column(j: int) -> np.ndarray:
        values = np.empty(n_values, dtype=np.float64)
        offset = 0
        for start in range(0, len(indices), chunk_windows):
            idx = indices[start : start + chunk_windows]
            block = np.asarray(X[idx, :, j], dtype=np.float64).reshape(-1)
            values[offset : offset + len(block)] = block
            offset += len(block)
        return values

    return _fit_from_column_loader(len(MODEL_FEATURE_COLUMNS), _load_column)


def _fit_from_column_loader(
    n_feat: int,
    load_column: Callable[[int], np.ndarray],
) -> FitTransformState:
    """Fit exact per-feature replacement and robust-scaling statistics.

    Only one float64 feature column is resident at a time.  Non-finite values
    are replaced from that training column before its median/IQR is measured.
    Sentinel ``-1`` values are excluded from both replacement and scaling
    statistics in the two documented sentinel columns.
    """

    train_median = np.empty(n_feat, dtype=np.float64)
    train_max_finite = np.empty(n_feat, dtype=np.float64)
    train_min_finite = np.empty(n_feat, dtype=np.float64)
    center = np.empty(n_feat, dtype=np.float64)
    scale = np.empty(n_feat, dtype=np.float64)

    for j in range(n_feat):
        col = load_column(j)
        # Exclude sentinels from statistics.
        if j in _SENTINEL_INDICES:
            finite_vals = col[np.isfinite(col) & (col != -1.0)]
        else:
            finite_vals = col[np.isfinite(col)]

        if len(finite_vals) == 0:
            # Degenerate column — use 0 as fallback.
            train_median[j] = 0.0
            train_max_finite[j] = 0.0
            train_min_finite[j] = 0.0
        else:
            train_median[j] = float(np.median(finite_vals))
            train_max_finite[j] = float(np.max(finite_vals))
            train_min_finite[j] = float(np.min(finite_vals))

        # Replace non-finite values with statistics from this same training
        # column, then compute exact RobustScaler-compatible quantiles.
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = train_median[j]
        posinf_mask = np.isposinf(col)
        if posinf_mask.any():
            col[posinf_mask] = train_max_finite[j]
        neginf_mask = np.isneginf(col)
        if neginf_mask.any():
            col[neginf_mask] = train_min_finite[j]

        scale_values = col[col != -1.0] if j in _SENTINEL_INDICES else col
        if len(scale_values) == 0:
            center[j] = 0.0
            scale[j] = 1.0
        else:
            q25, q50, q75 = np.percentile(scale_values, [25.0, 50.0, 75.0])
            center[j] = float(q50)
            width = float(q75 - q25)
            scale[j] = width if width > 0.0 else 1.0

    scaler = RobustScaler()
    scaler.center_ = center.copy()
    scaler.scale_ = scale.copy()
    scaler.n_features_in_ = n_feat

    return FitTransformState(
        feature_names=list(MODEL_FEATURE_COLUMNS),
        center=center,
        scale=scale,
        train_median=train_median,
        train_max_finite=train_max_finite,
        train_min_finite=train_min_finite,
        _scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def apply_transforms(
    X: np.ndarray,
    state: FitTransformState,
) -> np.ndarray:
    """Apply the fitted transforms to a split's feature tensor.

    Sentinel ``-1`` values in :data:`SENTINEL_NEGATIVE_COLUMNS` are preserved
    unchanged through a mask-and-restore pattern.

    Parameters
    ----------
    X:
        Float32 array of shape ``(n_windows, T, 78)``.
    state:
        Statistics returned by :func:`fit_transforms`.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape, scaled.
    """
    n_win, T, n_feat = X.shape
    flat = X.reshape(-1, n_feat).astype(np.float64)

    # Record sentinel positions before any transformation.
    sentinel_masks: List[np.ndarray] = []
    for j in _SENTINEL_INDICES:
        sentinel_masks.append(flat[:, j] == -1.0)

    # Safety-net finite replacement (train-derived statistics).
    flat = _replace_nonfinite(
        flat,
        state.train_median,
        state.train_max_finite,
        state.train_min_finite,
    )

    # Scale.
    flat_scaled = state._scaler.transform(flat)

    # Restore sentinels.
    for mask, j in zip(sentinel_masks, _SENTINEL_INDICES):
        flat_scaled[mask, j] = -1.0

    return flat_scaled.reshape(n_win, T, n_feat).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replace_nonfinite(
    arr: np.ndarray,
    median: np.ndarray,
    max_finite: np.ndarray,
    min_finite: np.ndarray,
) -> np.ndarray:
    """Replace NaN → median, +inf → max_finite, -inf → min_finite (in-place copy)."""
    out = arr.copy()
    for j in range(arr.shape[1]):
        col = out[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = median[j]
        posinf_mask = col == np.inf
        if posinf_mask.any():
            col[posinf_mask] = max_finite[j]
        neginf_mask = col == -np.inf
        if neginf_mask.any():
            col[neginf_mask] = min_finite[j]
        out[:, j] = col
    return out
