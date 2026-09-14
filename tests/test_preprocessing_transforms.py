"""Tests for src.preprocessing.transforms."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.schema import MODEL_FEATURE_COLUMNS, SENTINEL_NEGATIVE_COLUMNS
from src.preprocessing.transforms import (
    FitTransformState,
    _replace_nonfinite,
    apply_transforms,
    fit_transforms,
    fit_transforms_indexed,
)

T = 5
N_FEAT = 78
_SENTINEL_IDX = tuple(MODEL_FEATURE_COLUMNS.index(c) for c in SENTINEL_NEGATIVE_COLUMNS)


def _make_X(n_windows: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 100.0, size=(n_windows, T, N_FEAT)).astype(np.float32)


class TestReplaceNonfinite:
    def _stats(self, n: int):
        median = np.full(n, 5.0)
        max_f = np.full(n, 100.0)
        min_f = np.full(n, 0.0)
        return median, max_f, min_f

    def test_nan_replaced_with_median(self):
        arr = np.ones((4, 2), dtype=np.float64)
        arr[1, 0] = np.nan
        median, max_f, min_f = np.array([9.0, 9.0]), np.array([100.0, 100.0]), np.array([0.0, 0.0])
        out = _replace_nonfinite(arr, median, max_f, min_f)
        assert out[1, 0] == pytest.approx(9.0)
        assert out[0, 0] == pytest.approx(1.0)

    def test_posinf_replaced_with_max(self):
        arr = np.ones((3, 1), dtype=np.float64)
        arr[2, 0] = np.inf
        out = _replace_nonfinite(arr, np.array([5.0]), np.array([42.0]), np.array([0.0]))
        assert out[2, 0] == pytest.approx(42.0)

    def test_neginf_replaced_with_min(self):
        arr = np.ones((3, 1), dtype=np.float64)
        arr[0, 0] = -np.inf
        out = _replace_nonfinite(arr, np.array([5.0]), np.array([42.0]), np.array([-7.0]))
        assert out[0, 0] == pytest.approx(-7.0)

    def test_no_mutation_of_input(self):
        arr = np.array([[np.nan, 1.0]])
        orig = arr.copy()
        _replace_nonfinite(arr, np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([-1.0, -1.0]))
        np.testing.assert_array_equal(arr, orig)


class TestFitTransforms:
    def test_returns_fit_state(self):
        X = _make_X(20)
        state = fit_transforms(X)
        assert isinstance(state, FitTransformState)

    def test_feature_names_match_schema(self):
        X = _make_X(10)
        state = fit_transforms(X)
        assert state.feature_names == list(MODEL_FEATURE_COLUMNS)

    def test_center_scale_shapes(self):
        X = _make_X(10)
        state = fit_transforms(X)
        assert state.center.shape == (N_FEAT,)
        assert state.scale.shape == (N_FEAT,)

    def test_scaler_attached(self):
        X = _make_X(10)
        state = fit_transforms(X)
        assert state._scaler is not None

    def test_wrong_feature_count_raises(self):
        X = np.ones((10, T, 77), dtype=np.float32)
        with pytest.raises(AssertionError):
            fit_transforms(X)

    def test_sentinels_excluded_from_stats(self):
        """Columns with -1 sentinels must not contaminate the median."""
        X = _make_X(50)
        # Inject -1 sentinels into the first sentinel column.
        j = _SENTINEL_IDX[0]
        X[:10, :, j] = -1.0
        state = fit_transforms(X)
        # Median should be well above -1 since non-sentinel values are 0.1–100.
        assert state.train_median[j] > 0.0

    def test_indexed_fit_matches_materialized_fit(self):
        """The bounded E2 fit must equal fitting the selected tensor directly."""
        X = _make_X(30, seed=31)
        indices = np.array([0, 2, 3, 8, 13, 21, 29], dtype=np.int64)
        X[2, 1, 5] = np.nan
        X[8, 2, 6] = np.inf
        X[13, 3, 7] = -np.inf
        X[21, 4, _SENTINEL_IDX[0]] = -1.0

        expected = fit_transforms(X[indices])
        actual = fit_transforms_indexed(X, indices, chunk_windows=2)
        for name in (
            "center",
            "scale",
            "train_median",
            "train_max_finite",
            "train_min_finite",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))

    def test_indexed_fit_rejects_nonpositive_chunk_size(self):
        X = _make_X(3)
        with pytest.raises(ValueError, match="chunk_windows"):
            fit_transforms_indexed(X, np.array([0, 1]), chunk_windows=0)


class TestApplyTransforms:
    def test_output_shape(self):
        X_train = _make_X(20)
        state = fit_transforms(X_train)
        X_val = _make_X(8, seed=99)
        out = apply_transforms(X_val, state)
        assert out.shape == X_val.shape

    def test_output_dtype_float32(self):
        X = _make_X(10)
        state = fit_transforms(X)
        out = apply_transforms(X, state)
        assert out.dtype == np.float32

    def test_sentinel_values_preserved(self):
        """−1 in sentinel columns must survive scaling unchanged."""
        X_train = _make_X(50)
        state = fit_transforms(X_train)

        X_test = _make_X(4)
        j = _SENTINEL_IDX[0]
        X_test[0, 2, j] = -1.0   # inject a sentinel
        out = apply_transforms(X_test, state)
        assert out[0, 2, j] == pytest.approx(-1.0, abs=1e-6)

    def test_non_sentinel_values_changed(self):
        """Non-sentinel values must actually be scaled (not identity)."""
        X_train = _make_X(50)
        state = fit_transforms(X_train)
        X_test = _make_X(4, seed=7)
        out = apply_transforms(X_test, state)
        # The scaler shifts by median; if data ≠ median, values change.
        assert not np.allclose(out, X_test, atol=1e-3)

    def test_to_dict_round_trip(self):
        X = _make_X(10)
        state = fit_transforms(X)
        d = state.to_dict()
        state2 = FitTransformState.from_dict(d)
        np.testing.assert_allclose(state2.center, state.center)
        np.testing.assert_allclose(state2.scale, state.scale)

    def test_from_dict_scaler_works(self):
        X_train = _make_X(20)
        state = fit_transforms(X_train)
        d = state.to_dict()
        state2 = FitTransformState.from_dict(d)
        X_test = _make_X(4, seed=5)
        out1 = apply_transforms(X_test, state)
        out2 = apply_transforms(X_test, state2)
        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_nan_in_input_replaced(self):
        X_train = _make_X(20)
        state = fit_transforms(X_train)
        X_test = _make_X(4)
        X_test[1, 0, 5] = np.nan
        out = apply_transforms(X_test, state)
        assert np.isfinite(out).all()
