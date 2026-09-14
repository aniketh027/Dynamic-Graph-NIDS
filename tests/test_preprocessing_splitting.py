"""Tests for src.preprocessing.splitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.labels import (
    ABSENT_CLASSES,
    BINARY_ATTACK,
    BINARY_BENIGN,
    LABEL_TO_INT,
    LABEL_VOCABULARY,
)
from src.preprocessing.splitting import (
    ELIGIBILITY_MIN_TEST,
    ELIGIBILITY_MIN_TRAIN,
    ELIGIBILITY_MIN_VAL,
    SplitArrays,
    SplitResult,
    _class_distribution,
    _split_indices,
    class_availability_matrix,
    class_eligibility,
    experiment1_global_chronological,
    experiment2_per_class_chronological,
    experiment3_binary_chronological,
)

T = 5
N_FEAT = 78


def _make_Xy(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, T, N_FEAT)).astype(np.float32)
    # Alternate between two classes.
    codes = [LABEL_TO_INT["Benign"], LABEL_TO_INT["Bot"]]
    y = np.array([codes[i % 2] for i in range(n)], dtype=np.int16)
    return X, y


def _make_Xy_multiclass(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a dataset with one window per class × n_per_class."""
    rng = np.random.default_rng(42)
    labels = LABEL_VOCABULARY  # 15 classes
    X_parts, y_parts = [], []
    for lbl in labels:
        code = LABEL_TO_INT[lbl]
        X_parts.append(rng.uniform(0, 1, (n_per_class, T, N_FEAT)).astype(np.float32))
        y_parts.append(np.full(n_per_class, code, dtype=np.int16))
    return np.concatenate(X_parts), np.concatenate(y_parts)


class TestHelpers:
    def test_split_indices_basic(self):
        te, ve = _split_indices(100, 0.70, 0.10)
        assert te == 70
        assert ve == 80

    def test_split_indices_remainder_goes_to_test(self):
        te, ve = _split_indices(10, 0.70, 0.10)
        # 10*0.70=7, 10*0.10=1 → test = 2
        assert te == 7
        assert ve == 8

    def test_class_distribution_empty(self):
        assert _class_distribution(np.array([], dtype=np.int16)) == {}

    def test_class_distribution_counts(self):
        y = np.array([LABEL_TO_INT["Benign"]] * 3 + [LABEL_TO_INT["Bot"]] * 2, dtype=np.int16)
        dist = _class_distribution(y)
        assert dist["Benign"] == 3
        assert dist["Bot"] == 2


class TestExperiment1:
    def test_sizes_sum_to_total(self):
        X, y = _make_Xy(100)
        r = experiment1_global_chronological(X, y)
        a = r.arrays
        assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 100

    def test_no_shuffle_preserves_order(self):
        X, y = _make_Xy(100)
        r = experiment1_global_chronological(X, y, train_ratio=0.70, val_ratio=0.10)
        a = r.arrays
        # First train window must equal X[0].
        np.testing.assert_array_equal(a.X_train[0], X[0])
        # First val window must equal X[70].
        np.testing.assert_array_equal(a.X_val[0], X[70])
        # First test window must equal X[80].
        np.testing.assert_array_equal(a.X_test[0], X[80])

    def test_metadata_experiment_key(self):
        X, y = _make_Xy(50)
        r = experiment1_global_chronological(X, y)
        assert r.metadata["experiment"] == "1_global_chronological"

    def test_metadata_counts(self):
        X, y = _make_Xy(100)
        r = experiment1_global_chronological(X, y)
        assert r.metadata["n_train"] == len(r.arrays.y_train)
        assert r.metadata["n_val"] == len(r.arrays.y_val)
        assert r.metadata["n_test"] == len(r.arrays.y_test)

    def test_rare_class_detection(self):
        # Make a tiny dataset so no class gets ≥ 30 test windows.
        X, y = _make_Xy(20)
        r = experiment1_global_chronological(X, y, rare_class_min_test_windows=30)
        # With only ~4 test windows split between 2 classes, both are rare.
        assert len(r.metadata["rare_classes"]) > 0

    def test_no_rare_classes_when_sufficient(self):
        # 1000 windows, alternating Benign/Bot.
        X, y = _make_Xy(1000)
        # With 200 test windows and 2 classes ~100 each, threshold of 30 → no rare.
        r = experiment1_global_chronological(X, y, rare_class_min_test_windows=30)
        assert len(r.metadata["rare_classes"]) == 0

    def test_X_shapes(self):
        X, y = _make_Xy(50)
        r = experiment1_global_chronological(X, y)
        a = r.arrays
        assert a.X_train.ndim == 3
        assert a.X_train.shape[1] == T
        assert a.X_train.shape[2] == N_FEAT


class TestExperiment2:
    def test_sizes_sum_to_total(self):
        X, y = _make_Xy(100)
        r = experiment2_per_class_chronological(X, y)
        a = r.arrays
        assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 100

    def test_metadata_experiment_key(self):
        X, y = _make_Xy(50)
        r = experiment2_per_class_chronological(X, y)
        assert r.metadata["experiment"] == "2_per_class_chronological"

    def test_per_class_counts_present(self):
        X, y = _make_Xy(100)
        r = experiment2_per_class_chronological(X, y)
        pcc = r.metadata["per_class_split_counts"]
        assert "Benign" in pcc
        assert "Bot" in pcc

    def test_absent_class_flagged(self):
        X, y = _make_Xy(50)
        r = experiment2_per_class_chronological(X, y)
        pcc = r.metadata["per_class_split_counts"]
        # SSH-Bruteforce is absent from our synthetic data.
        assert pcc["SSH-Bruteforce"]["absent"] is True

    def test_not_described_as_globally_chronological_or_leakage_free(self):
        """E2 must explicitly record that it is neither of those things."""
        X, y = _make_Xy(50)
        r = experiment2_per_class_chronological(X, y)
        assert r.metadata["globally_chronological"] is False
        assert r.metadata["leakage_free"] is False
        assert r.metadata["chronological_scope"] == "within each class only"
        caveat = r.metadata["temporal_isolation_caveat"].lower()
        assert "does not provide strict global chronological isolation" in caveat
        assert "must not be described as leakage-free" in caveat

    def test_class_order_preserved_within_class(self):
        """Each class's windows appear in global-index order in the test set."""
        n = 200
        X, y = _make_Xy(n)
        r = experiment2_per_class_chronological(X, y)
        # For Benign windows: find their positions in the test set.
        benign_code = LABEL_TO_INT["Benign"]
        test_benign_mask = r.arrays.y_test == benign_code
        # The test arrays are sorted by global index, so within-class we expect
        # the global indices to be increasing (by construction in _pool).
        # We verify by checking the X values match expected positions.
        # Even positions in y are Benign (0, 2, 4 ...).
        even_positions = np.arange(0, n, 2)
        # The last 20% of even_positions should be in test.
        expected_test_start = int(np.floor(len(even_positions) * 0.80))
        expected_test_benign_idx = even_positions[expected_test_start:]
        assert test_benign_mask.sum() == len(expected_test_benign_idx)

    def test_multiclass_split(self):
        X, y = _make_Xy_multiclass(100)
        r = experiment2_per_class_chronological(X, y, rare_class_min_test_windows=10)
        a = r.arrays
        total = len(a.y_train) + len(a.y_val) + len(a.y_test)
        assert total == len(y)

    def test_rare_class_threshold(self):
        # With only 10 windows per class, most classes have <3 test windows.
        X, y = _make_Xy_multiclass(10)
        r = experiment2_per_class_chronological(X, y, rare_class_min_test_windows=5)
        assert len(r.metadata["rare_classes"]) > 0

    def test_marked_as_primary_experiment_a(self):
        X, y = _make_Xy(50)
        r = experiment2_per_class_chronological(X, y)
        assert r.metadata["primary_experiment"] == "A"
        assert r.metadata["is_primary_benchmark"] is True
        assert r.metadata["role"] == "multiclass separability"

    def test_eligibility_table_present(self):
        X, y = _make_Xy_multiclass(200)
        r = experiment2_per_class_chronological(X, y)
        el = r.metadata["class_eligibility"]
        assert el["thresholds"]["min_train"] == ELIGIBILITY_MIN_TRAIN
        # 200 per class → 140 train / 20 val / 40 test → val fails the ≥30 rule.
        assert el["n_eligible"] == 0
        for cls in LABEL_VOCABULARY:
            reason = el["per_class"][cls]["ineligibility_reason"]
            assert "validation windows 20 < required 30" in reason


# ---------------------------------------------------------------------------
# §1 — E1 framing and the class availability matrix
# ---------------------------------------------------------------------------

class TestExperiment1Framing:
    def test_marked_as_diagnostic_not_benchmark(self):
        X, y = _make_Xy(50)
        r = experiment1_global_chronological(X, y)
        assert r.metadata["is_primary_benchmark"] is False
        assert r.metadata["role"] == (
            "strict chronological multiclass generalization diagnostic"
        )
        assert r.metadata["globally_chronological"] is True
        assert r.metadata["split_index_overlap"] is False
        assert r.metadata["leakage_free"] is False

    def test_explicitly_not_zero_shot_anomaly_detection(self):
        X, y = _make_Xy(50)
        r = experiment1_global_chronological(X, y)
        note = r.metadata["not_zero_shot_anomaly_detection"]
        assert "must not be labelled 'zero-shot anomaly detection'" in note

    def test_availability_matrix_marks_unseen_classes(self):
        """A class confined to the tail of the sequence is unseen in training."""
        n = 100
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (n, T, N_FEAT)).astype(np.float32)
        y = np.full(n, LABEL_TO_INT["Benign"], dtype=np.int16)
        y[90:] = LABEL_TO_INT["Bot"]      # only in the final 10% → test only

        r = experiment1_global_chronological(X, y)
        av = r.metadata["class_availability_matrix"]
        assert "Bot" in av["classes_unseen_during_training"]
        assert av["per_class"]["Bot"]["in_train"] is False
        assert av["per_class"]["Bot"]["in_test"] is True
        assert av["per_class"]["Bot"][
            "supervised_per_class_evaluation_possible"
        ] is False
        assert av["per_class"]["Bot"]["status"] == (
            "present in test but UNSEEN during training"
        )
        assert "Benign" in av["classes_with_possible_supervised_evaluation"]

    def test_availability_matrix_covers_whole_vocabulary(self):
        X, y = _make_Xy(50)
        r = experiment1_global_chronological(X, y)
        av = r.metadata["class_availability_matrix"]
        for cls in LABEL_VOCABULARY:
            assert cls in av["per_class"]
        # Absent-from-corpus classes are annotated, not silently omitted.
        for cls in ABSENT_CLASSES:
            assert av["per_class"][cls]["status"] == (
                "not evaluated — absent from corpus"
            )

    def test_availability_counts_match_distributions(self):
        X, y = _make_Xy_multiclass(50)
        r = experiment1_global_chronological(X, y)
        av = r.metadata["class_availability_matrix"]
        for cls, row in r.metadata["train_class_distribution"].items():
            assert av["per_class"][cls]["n_train"] == row


# ---------------------------------------------------------------------------
# §2 — class eligibility
# ---------------------------------------------------------------------------

class TestClassEligibility:
    def _y(self, counts: dict) -> tuple:
        """Build (y_train, y_val, y_test) from {class: (n_tr, n_va, n_te)}."""
        parts = ([], [], [])
        for cls, (a, b, c) in counts.items():
            code = LABEL_TO_INT[cls]
            for bucket, n in zip(parts, (a, b, c)):
                bucket.extend([code] * n)
        return tuple(np.array(p, dtype=np.int16) for p in parts)

    def test_class_at_exact_thresholds_is_eligible(self):
        y_tr, y_va, y_te = self._y({
            "Benign": (ELIGIBILITY_MIN_TRAIN, ELIGIBILITY_MIN_VAL, ELIGIBILITY_MIN_TEST),
        })
        el = class_eligibility(y_tr, y_va, y_te)
        assert el["per_class"]["Benign"]["eligible"] is True
        assert el["per_class"]["Benign"]["ineligibility_reasons"] == []
        assert "Benign" in el["eligible_classes"]

    def test_one_short_of_each_threshold_is_ineligible(self):
        for short in ("train", "val", "test"):
            counts = [ELIGIBILITY_MIN_TRAIN, ELIGIBILITY_MIN_VAL, ELIGIBILITY_MIN_TEST]
            idx = {"train": 0, "val": 1, "test": 2}[short]
            counts[idx] -= 1
            y_tr, y_va, y_te = self._y({"Bot": tuple(counts)})
            el = class_eligibility(y_tr, y_va, y_te)
            assert el["per_class"]["Bot"]["eligible"] is False, short
            assert short.replace("val", "validation") in \
                el["per_class"]["Bot"]["ineligibility_reason"]

    def test_reason_records_every_failing_criterion(self):
        y_tr, y_va, y_te = self._y({"Bot": (1, 1, 1)})
        el = class_eligibility(y_tr, y_va, y_te)
        reasons = el["per_class"]["Bot"]["ineligibility_reasons"]
        assert len(reasons) == 3
        assert any("train windows 1" in r for r in reasons)
        assert any("validation windows 1" in r for r in reasons)
        assert any("test windows 1" in r for r in reasons)

    def test_absent_class_reason(self):
        y_tr, y_va, y_te = self._y({"Benign": (200, 50, 50)})
        el = class_eligibility(y_tr, y_va, y_te)
        assert el["per_class"]["Bot"]["n_total"] == 0
        assert el["per_class"]["Bot"]["ineligibility_reason"] == (
            "absent from the windowed corpus (0 windows)"
        )

    def test_every_vocabulary_class_appears_exactly_once(self):
        y_tr, y_va, y_te = self._y({"Benign": (200, 50, 50)})
        el = class_eligibility(y_tr, y_va, y_te)
        for cls in LABEL_VOCABULARY:
            assert cls in el["per_class"]
        names = el["eligible_classes"] + el["ineligible_classes"]
        for cls in LABEL_VOCABULARY:
            assert names.count(cls) == 1

    def test_counts_are_recorded_for_ineligible_classes(self):
        """Ineligible classes keep their counts in the report, not dropped."""
        y_tr, y_va, y_te = self._y({"Benign": (200, 50, 50), "Bot": (7, 2, 1)})
        el = class_eligibility(y_tr, y_va, y_te)
        row = el["per_class"]["Bot"]
        assert (row["n_train"], row["n_val"], row["n_test"]) == (7, 2, 1)
        assert row["n_total"] == 10

    def test_policy_note_disclaims_statistical_significance(self):
        el = class_eligibility(*self._y({"Benign": (200, 50, 50)}))
        note = el["policy_note"]
        assert "ENGINEERING REPORTING POLICY" in note
        assert "NOT a claim of statistical significance" in note
        assert "NOT a universal statistical requirement" in note

    def test_no_macro_class_merging(self):
        el = class_eligibility(*self._y({"Benign": (200, 50, 50)}))
        assert "not merged into semantic macro-classes" in el["rare_class_handling"]

    def test_custom_thresholds_respected(self):
        y_tr, y_va, y_te = self._y({"Bot": (10, 5, 5)})
        el = class_eligibility(y_tr, y_va, y_te, min_train=10, min_val=5, min_test=5)
        assert el["per_class"]["Bot"]["eligible"] is True


# ---------------------------------------------------------------------------
# §1/§6 — Experiment 3: strict chronological binary
# ---------------------------------------------------------------------------

class TestExperiment3:
    def test_sizes_sum_to_total(self):
        X, y = _make_Xy(100)
        r = experiment3_binary_chronological(X, y)
        a = r.arrays
        assert len(a.y_train) + len(a.y_val) + len(a.y_test) == 100

    def test_same_boundaries_as_experiment1(self):
        X, y = _make_Xy_multiclass(100)
        r1 = experiment1_global_chronological(X, y)
        r3 = experiment3_binary_chronological(X, y)
        assert r3.metadata["split_boundaries"] == r1.metadata["split_boundaries"]
        assert r3.metadata["n_train"] == r1.metadata["n_train"]
        assert r3.metadata["n_val"] == r1.metadata["n_val"]
        assert r3.metadata["n_test"] == r1.metadata["n_test"]

    def test_features_identical_to_experiment1(self):
        """Only the target changes; the window tensor is untouched."""
        X, y = _make_Xy_multiclass(40)
        r1 = experiment1_global_chronological(X, y)
        r3 = experiment3_binary_chronological(X, y)
        np.testing.assert_array_equal(r1.arrays.X_train, r3.arrays.X_train)
        np.testing.assert_array_equal(r1.arrays.X_test, r3.arrays.X_test)

    def test_target_is_binary(self):
        X, y = _make_Xy_multiclass(50)
        r = experiment3_binary_chronological(X, y)
        for arr in (r.arrays.y_train, r.arrays.y_val, r.arrays.y_test):
            assert set(np.unique(arr).tolist()) <= {BINARY_BENIGN, BINARY_ATTACK}

    def test_benign_maps_to_zero_attacks_to_one(self):
        X, y = _make_Xy_multiclass(50)
        r = experiment3_binary_chronological(X, y)
        y_bin = np.concatenate([r.arrays.y_train, r.arrays.y_val, r.arrays.y_test])
        benign_code = LABEL_TO_INT["Benign"]
        assert np.all(y_bin[y == benign_code] == BINARY_BENIGN)
        assert np.all(y_bin[y != benign_code] == BINARY_ATTACK)

    def test_ordering_preserved(self):
        """No shuffling: the concatenated binary target matches the input order."""
        X, y = _make_Xy_multiclass(30)
        r = experiment3_binary_chronological(X, y)
        y_bin = np.concatenate([r.arrays.y_train, r.arrays.y_val, r.arrays.y_test])
        expected = np.where(y == LABEL_TO_INT["Benign"], BINARY_BENIGN, BINARY_ATTACK)
        np.testing.assert_array_equal(y_bin, expected)

    def test_marked_as_primary_experiment_b(self):
        X, y = _make_Xy(50)
        r = experiment3_binary_chronological(X, y)
        assert r.metadata["experiment"] == "3_binary_chronological"
        assert r.metadata["primary_experiment"] == "B"
        assert r.metadata["is_primary_benchmark"] is True
        assert r.metadata["globally_chronological"] is True
        assert r.metadata["split_index_overlap"] is False
        assert r.metadata["leakage_free"] is False

    def test_binary_distributions_reported(self):
        X, y = _make_Xy(100)   # alternating Benign / Bot
        r = experiment3_binary_chronological(X, y)
        tr = r.metadata["train_binary_distribution"]
        assert set(tr) == {"Benign", "Attack"}
        assert tr["Benign"] + tr["Attack"] == r.metadata["n_train"]
        assert r.metadata["both_binary_classes_present_in_all_splits"] is True

    def test_binary_class_availability_is_measured_not_assumed(self):
        X = np.zeros((20, T, N_FEAT), dtype=np.float32)
        y = np.full(20, LABEL_TO_INT["Benign"], dtype=np.int16)
        r = experiment3_binary_chronological(X, y)
        assert r.metadata["both_binary_classes_present_in_all_splits"] is False

    def test_multiclass_composition_kept_for_diagnostics(self):
        X, y = _make_Xy(100)
        r = experiment3_binary_chronological(X, y)
        comp = r.metadata["train_multiclass_composition"]
        assert "Benign" in comp and "Bot" in comp
        assert sum(comp.values()) == r.metadata["n_train"]

    def test_binary_mapping_embedded_in_metadata(self):
        X, y = _make_Xy(50)
        r = experiment3_binary_chronological(X, y)
        bm = r.metadata["binary_mapping"]
        assert bm["n_classes"] == 2
        assert bm["n_attack_labels"] == 14
        assert len(bm["label_to_binary"]) == len(LABEL_VOCABULARY)


class TestAvailabilityMatrixDirect:
    def test_absent_class_status(self):
        y = np.array([LABEL_TO_INT["Benign"]] * 10, dtype=np.int16)
        av = class_availability_matrix(y, y, y)
        assert av["per_class"]["Bot"]["status"] == "absent from the windowed corpus"
        assert "Bot" not in av["classes_unseen_during_training"]

    def test_trained_but_absent_from_test(self):
        tr = np.array([LABEL_TO_INT["Bot"]] * 5, dtype=np.int16)
        empty = np.array([], dtype=np.int16)
        av = class_availability_matrix(tr, empty, empty)
        assert av["per_class"]["Bot"]["status"] == "trained but absent from test"
        assert "Bot" in av["classes_absent_from_test"]
