"""Train / Validation / Test split construction for Stage C.

Three clearly separated configurations are implemented.  They answer different
questions and must not be conflated in reporting.

Experiment 1 — Strict Chronological Multiclass Diagnostic
    The full global window sequence (capture-order) is cut at the 70/10/20
    boundaries.  No shuffling.  Every window's relative temporal position is
    preserved and split indices do not overlap.  This alone is not called a
    "leakage-free multiclass evaluation": retained temporal-duplicate feature
    states can recur across boundaries, and the class-coverage failure makes
    the protocol structurally unsuitable as the primary multiclass benchmark.

    E1 is a **strict chronological multiclass generalization diagnostic**.  It
    is NOT a primary supervised multiclass performance benchmark: because the
    attack classes in this corpus are concentrated in particular capture days,
    a strict global chronological cut leaves several classes with no training
    windows at all.  Supervised per-class evaluation is impossible for those
    classes — a classifier cannot be asked to name a class it was never shown.

    E1 is also NOT "zero-shot anomaly detection".  Nothing here trains an
    anomaly detector, and the unseen classes are an artefact of the corpus
    schedule rather than a designed open-set protocol.

    :func:`experiment1_global_chronological` therefore reports a class
    train/validation/test **availability matrix** so the reader can see exactly
    which classes are unseen during training.

Experiment 2 — Multiclass Separability  (PRIMARY EXPERIMENT A)
    Within each class the windows are sorted by their position in the global
    sequence (the E1 window index), then split 70/10/20 independently.  The
    per-class splits are pooled: train = union of all classes' train portions,
    and so on.  Chronological order is preserved *within* each class.

    This is the primary supervised multiclass experiment.  It does NOT provide
    strict global chronological isolation across classes: a train window of one
    class may be later in wall-clock time than a test window of another class.
    E2 must never be described as leakage-free or globally chronological.

    Final locked-test reporting for E2 is restricted to *reporting-eligible*
    classes — see :func:`class_eligibility`.  Train/validation decisions use
    the separate test-blind :func:`validation_selection_eligibility` policy.

Experiment 3 — Strict Chronological Binary Classification  (PRIMARY EXPERIMENT B)
    Same global window sequence and the same strict 70/10/20 chronological cut
    as E1, but the target is binary: ``Benign -> 0``, every non-Benign label
    ``-> 1``.  Original row/window ordering is preserved; the raw data is not
    altered.

    Collapsing to binary removes E1's structural problem — the model is never
    required to identify an attack subclass it has not seen — so E3 measures
    temporal generalization under a strict chronological protocol.

Validation-selection eligibility (E2 train/validation decisions)
    A class enters the validation objective when it has at least 100 train and
    30 validation windows.  Test labels, support, membership, and metadata are
    not inputs to this policy.

Final-reporting eligibility (E2 locked-test reporting)
    Post-windowing, a class is eligible if it has at least 100 train, 30
    validation and 30 test windows.  This is an ENGINEERING REPORTING POLICY,
    not a claim of statistical significance and not a universal statistical
    requirement.  Ineligible classes are excluded from the eligible-class
    evaluation set but retained in the dataset with their full counts, and
    their labels are never modified or merged.

Outputs
    All three experiments share the ``SplitResult`` type: NumPy arrays
    ``X_train / y_train``, ``X_val / y_val``, ``X_test / y_test`` and a
    metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.preprocessing.labels import (
    ABSENT_CLASSES,
    BINARY_INT_TO_NAME,
    INT_TO_LABEL,
    LABEL_TO_INT,
    LABEL_VOCABULARY,
    binary_codes_from_multiclass,
    binary_mapping_dict,
)

# ---------------------------------------------------------------------------
# E2 validation-selection and final-reporting policies
# ---------------------------------------------------------------------------

#: Minimum post-windowing train windows for a class to be eligible.
ELIGIBILITY_MIN_TRAIN: int = 100
#: Minimum post-windowing validation windows for a class to be eligible.
ELIGIBILITY_MIN_VAL: int = 30
#: Minimum post-windowing test windows for a class to be eligible.
ELIGIBILITY_MIN_TEST: int = 30

VALIDATION_SELECTION_POLICY_NOTE: str = (
    "A class is eligible for E2 validation selection when it has at least "
    f"{ELIGIBILITY_MIN_TRAIN} train and {ELIGIBILITY_MIN_VAL} validation "
    "windows after windowing.  Test labels, support, membership, metadata, "
    "and final-reporting eligibility are not consulted."
)

ELIGIBILITY_POLICY_NOTE: str = (
    "A class is eligible for primary multiclass reporting when it has at least "
    f"{ELIGIBILITY_MIN_TRAIN} train, {ELIGIBILITY_MIN_VAL} validation and "
    f"{ELIGIBILITY_MIN_TEST} test windows after windowing.  This threshold is "
    "an ENGINEERING REPORTING POLICY chosen so that per-class metrics are not "
    "computed over a handful of windows.  It is NOT a claim of statistical "
    "significance and NOT a universal statistical requirement.  Ineligible "
    "classes are excluded from the eligible-class evaluation set but are "
    "retained in the dataset with their full counts; their labels are never "
    "modified, merged into macro-classes, or dropped."
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class SplitArrays:
    """Six arrays for one experiment's train/val/test split."""

    X_train: np.ndarray   # (n_train, T, 78) float32
    y_train: np.ndarray   # (n_train,) int16
    X_val: np.ndarray     # (n_val,   T, 78) float32
    y_val: np.ndarray     # (n_val,)   int16
    X_test: np.ndarray    # (n_test,   T, 78) float32
    y_test: np.ndarray    # (n_test,)  int16


@dataclass
class SplitResult:
    """Output of one experiment's split."""

    arrays: SplitArrays
    metadata: Dict[str, object]


@dataclass
class IndexSplitArrays:
    """Window-index membership for a train/validation/test split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class IndexSplitResult:
    """Index-only split output used by the memory-safe full pipeline."""

    indices: IndexSplitArrays
    metadata: Dict[str, object]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_distribution(y: np.ndarray) -> Dict[str, int]:
    """Return label_string → window count for a multiclass label array."""
    dist: Dict[str, int] = {}
    if len(y) == 0:
        return dist
    codes, counts = np.unique(y, return_counts=True)
    for code, cnt in zip(codes, counts):
        dist[INT_TO_LABEL[int(code)]] = int(cnt)
    return dist


def _binary_distribution(y: np.ndarray) -> Dict[str, int]:
    """Return "Benign"/"Attack" → window count for a binary label array."""
    dist: Dict[str, int] = {name: 0 for name in BINARY_INT_TO_NAME.values()}
    if len(y) == 0:
        return dist
    codes, counts = np.unique(y, return_counts=True)
    for code, cnt in zip(codes, counts):
        dist[BINARY_INT_TO_NAME[int(code)]] = int(cnt)
    return dist


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    """Return (train_end, val_end) index boundaries for a sequence of length n."""
    train_end = int(np.floor(n * train_ratio))
    val_end = train_end + int(np.floor(n * val_ratio))
    return train_end, val_end


def class_availability_matrix(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """Report which classes are present in which split.

    Answers the question E1 forces: for which classes is supervised per-class
    evaluation even possible?  A class with zero train windows was never shown
    to the model, so a per-class supervised score for it is not interpretable
    regardless of what the classifier outputs.

    Returns a dict with a per-class table plus the derived class lists.
    """
    tr = _class_distribution(y_train)
    va = _class_distribution(y_val)
    te = _class_distribution(y_test)

    table: Dict[str, Dict[str, object]] = {}
    unseen_in_train: List[str] = []
    absent_from_test: List[str] = []
    evaluable: List[str] = []

    for cls in LABEL_VOCABULARY:
        n_tr, n_va, n_te = tr.get(cls, 0), va.get(cls, 0), te.get(cls, 0)
        total = n_tr + n_va + n_te
        in_train, in_test = n_tr > 0, n_te > 0

        if total == 0:
            status = "absent from the windowed corpus"
        elif not in_train and in_test:
            status = "present in test but UNSEEN during training"
        elif not in_train and not in_test:
            status = "present only in validation"
        elif in_train and not in_test:
            status = "trained but absent from test"
        else:
            status = "present in train and test"

        table[cls] = {
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "in_train": in_train,
            "in_val": n_va > 0,
            "in_test": in_test,
            "supervised_per_class_evaluation_possible": bool(in_train and in_test),
            "status": status,
        }

        if total > 0 and not in_train:
            unseen_in_train.append(cls)
        if total > 0 and not in_test:
            absent_from_test.append(cls)
        if in_train and in_test:
            evaluable.append(cls)

    for cls in ABSENT_CLASSES:
        table[cls] = {
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "in_train": False,
            "in_val": False,
            "in_test": False,
            "supervised_per_class_evaluation_possible": False,
            "status": "not evaluated — absent from corpus",
        }

    return {
        "per_class": table,
        "classes_unseen_during_training": unseen_in_train,
        "classes_absent_from_test": absent_from_test,
        "classes_with_possible_supervised_evaluation": evaluable,
        "n_classes_unseen_during_training": len(unseen_in_train),
        "note": (
            "Classes listed under classes_unseen_during_training have zero "
            "training windows under this split.  Per-class supervised "
            "evaluation is impossible for them: the model was never shown the "
            "class, so any per-class score is an artefact of the split, not a "
            "measurement of the model.  This is a property of the corpus "
            "capture schedule combined with a strict global chronological cut."
        ),
    }


def class_eligibility(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    min_train: int = ELIGIBILITY_MIN_TRAIN,
    min_val: int = ELIGIBILITY_MIN_VAL,
    min_test: int = ELIGIBILITY_MIN_TEST,
) -> Dict[str, object]:
    """Apply the post-windowing final-reporting eligibility rule.

    For every class in the vocabulary, record its train/val/test window counts,
    whether it is eligible, and — when it is not — the exact reason(s).  No
    class is merged, relabelled or removed from the dataset by this function;
    it only decides what enters the later locked-test reporting set.  This
    three-split function must not be used for train/validation decisions.
    """
    tr = _class_distribution(y_train)
    va = _class_distribution(y_val)
    te = _class_distribution(y_test)

    table: Dict[str, Dict[str, object]] = {}
    eligible: List[str] = []
    ineligible: List[str] = []

    for cls in LABEL_VOCABULARY:
        n_tr, n_va, n_te = tr.get(cls, 0), va.get(cls, 0), te.get(cls, 0)
        reasons: List[str] = []

        if n_tr + n_va + n_te == 0:
            reasons.append("absent from the windowed corpus (0 windows)")
        else:
            if n_tr < min_train:
                reasons.append(f"train windows {n_tr} < required {min_train}")
            if n_va < min_val:
                reasons.append(f"validation windows {n_va} < required {min_val}")
            if n_te < min_test:
                reasons.append(f"test windows {n_te} < required {min_test}")

        is_eligible = not reasons
        table[cls] = {
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "n_total": n_tr + n_va + n_te,
            "eligible": is_eligible,
            "ineligibility_reasons": reasons,
            "ineligibility_reason": "; ".join(reasons) if reasons else None,
        }
        (eligible if is_eligible else ineligible).append(cls)

    for cls in ABSENT_CLASSES:
        table[cls] = {
            "n_train": 0,
            "n_val": 0,
            "n_test": 0,
            "n_total": 0,
            "eligible": False,
            "ineligibility_reasons": ["not evaluated — absent from corpus"],
            "ineligibility_reason": "not evaluated — absent from corpus",
        }
        ineligible.append(cls)

    return {
        "thresholds": {
            "min_train": min_train,
            "min_val": min_val,
            "min_test": min_test,
        },
        "policy_note": ELIGIBILITY_POLICY_NOTE,
        "per_class": table,
        "eligible_classes": eligible,
        "ineligible_classes": ineligible,
        "n_eligible": len(eligible),
        "n_ineligible": len(ineligible),
        "eligible_class_codes": [LABEL_TO_INT[c] for c in eligible],
        "rare_class_handling": (
            "Ineligible classes keep their original labels and remain in all "
            "three splits and in the dataset report.  They are not merged into "
            "semantic macro-classes and their underlying labels are not "
            "modified in the raw or preprocessed data."
        ),
    }


def validation_selection_eligibility(
    y_train: np.ndarray,
    y_val: np.ndarray,
    min_train: int = ELIGIBILITY_MIN_TRAIN,
    min_val: int = ELIGIBILITY_MIN_VAL,
    *,
    chunk_size: int = 1_000_000,
) -> Dict[str, object]:
    """Derive the authoritative E2 validation-selection class set.

    The API intentionally accepts only train and validation targets.  Counts
    are accumulated in bounded chunks so production callers can pass the
    frozen target memory maps without materialising either split.
    """
    if not isinstance(min_train, int) or min_train < 1:
        raise ValueError("min_train must be an int >= 1")
    if not isinstance(min_val, int) or min_val < 1:
        raise ValueError("min_val must be an int >= 1")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be an int >= 1")

    def _bounded_counts(values: np.ndarray, split: str) -> np.ndarray:
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError(f"{split} targets must be a one-dimensional NumPy array")
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{split} targets must have an integer dtype")
        counts = np.zeros(len(LABEL_VOCABULARY), dtype=np.int64)
        for start in range(0, len(values), chunk_size):
            codes, observed = np.unique(
                values[start : start + chunk_size], return_counts=True
            )
            for raw_code, raw_count in zip(codes, observed, strict=True):
                code = int(raw_code)
                if code not in INT_TO_LABEL:
                    raise ValueError(
                        f"{split} targets contain non-canonical class code {code}"
                    )
                counts[code] += int(raw_count)
        return counts

    train_counts = _bounded_counts(y_train, "train")
    val_counts = _bounded_counts(y_val, "validation")
    selected_codes = [
        code
        for code in sorted(INT_TO_LABEL)
        if int(train_counts[code]) >= min_train
        and int(val_counts[code]) >= min_val
    ]
    if not selected_codes:
        raise ValueError("no classes satisfy E2 validation-selection thresholds")

    return {
        "thresholds": {"min_train": min_train, "min_val": min_val},
        "policy_note": VALIDATION_SELECTION_POLICY_NOTE,
        "test_support_consulted": False,
        "validation_selection_class_codes": selected_codes,
        "per_class": {
            INT_TO_LABEL[code]: {
                "n_train": int(train_counts[code]),
                "n_val": int(val_counts[code]),
                "selected": code in selected_codes,
            }
            for code in sorted(INT_TO_LABEL)
        },
    }


# ---------------------------------------------------------------------------
# Experiment 1 — strict chronological multiclass diagnostic
# ---------------------------------------------------------------------------

def experiment1_global_chronological(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    rare_class_min_test_windows: int = 30,
) -> SplitResult:
    """Strict global chronological 70 / 10 / 20 split (multiclass diagnostic).

    The split itself is unchanged from the original Stage C design: a sequential
    cut of the globally-ordered window sequence, no shuffling.  What changed is
    the framing recorded in the metadata — E1 is a diagnostic, not a primary
    supervised benchmark, and the class availability matrix now makes the
    unseen-class problem explicit rather than leaving it to be inferred from a
    confusion matrix.

    Parameters
    ----------
    X:
        Feature tensor, shape (n_windows, T, 78), float32.
    y:
        Multiclass label vector, shape (n_windows,), int16.
    train_ratio, val_ratio:
        Fractions that define the cut points.  Test gets the remainder.
    rare_class_min_test_windows:
        Classes with fewer test windows than this value are flagged as rare.
    """
    n = len(y)
    train_end, val_end = _split_indices(n, train_ratio, val_ratio)

    # Basic slicing → views, so no copy of the window tensor is made here.
    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    # Rare-class detection: only classes present in the full dataset but with
    # fewer than rare_class_min_test_windows windows in the test split.
    # Classes absent from the dataset entirely are not flagged here.
    present_classes = set(_class_distribution(y).keys())
    test_dist = _class_distribution(y_test)
    rare = [
        cls for cls in LABEL_VOCABULARY
        if cls in present_classes and test_dist.get(cls, 0) < rare_class_min_test_windows
    ]

    availability = class_availability_matrix(y_train, y_val, y_test)

    metadata: Dict[str, object] = {
        "experiment": "1_global_chronological",
        "role": "strict chronological multiclass generalization diagnostic",
        "is_primary_benchmark": False,
        "target": "multiclass (15-label vocabulary)",
        "description": (
            "Sequential 70/10/20 split on the globally-ordered T=5 window "
            "sequence.  No shuffling.  The relative temporal order of every "
            "window is preserved."
        ),
        "interpretation": (
            "E1 measures how a model trained strictly on the past behaves on "
            "strictly later traffic.  It is a diagnostic, NOT a primary "
            "supervised multiclass performance benchmark: several classes have "
            "zero training windows under this split, so supervised per-class "
            "evaluation is impossible for them and aggregate multiclass scores "
            "are dominated by that structural fact rather than by model "
            "quality.  See class_availability_matrix."
        ),
        "not_zero_shot_anomaly_detection": (
            "E1 must not be labelled 'zero-shot anomaly detection'.  No "
            "anomaly detector is trained or evaluated here, and the unseen "
            "classes are a by-product of the corpus capture schedule rather "
            "than a designed open-set protocol."
        ),
        "globally_chronological": True,
        "split_index_overlap": False,
        "leakage_free": False,
        "leakage_caveat": (
            "The split is strictly ordered and its window indices do not "
            "overlap, but E1 is not claimed to be a leakage-free multiclass "
            "evaluation. Retained temporal-duplicate feature/label states can "
            "recur across split boundaries."
        ),
        "n_windows_total": n,
        "split_boundaries": {
            "train_end_idx": train_end,
            "val_end_idx": val_end,
        },
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_class_distribution": _class_distribution(y_train),
        "val_class_distribution": _class_distribution(y_val),
        "test_class_distribution": test_dist,
        "class_availability_matrix": availability,
        "rare_class_min_test_windows": rare_class_min_test_windows,
        "rare_classes": rare,
        "rare_class_note": (
            "Classes listed here have fewer than rare_class_min_test_windows "
            "test windows.  They are retained in all splits but excluded from "
            "primary per-class performance claims.  They are never silently "
            "merged into another class."
        ),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 10),
    }

    return SplitResult(
        arrays=SplitArrays(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Experiment 2 — multiclass separability (PRIMARY EXPERIMENT A)
# ---------------------------------------------------------------------------

def experiment2_per_class_chronological(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    rare_class_min_test_windows: int = 30,
    min_train: int = ELIGIBILITY_MIN_TRAIN,
    min_val: int = ELIGIBILITY_MIN_VAL,
    min_test: int = ELIGIBILITY_MIN_TEST,
) -> SplitResult:
    """Per-class chronological 70 / 10 / 20 split — PRIMARY EXPERIMENT A.

    Within each class, windows are sorted by their global index (their position
    in the E1 sequence), then split 70/10/20.  The per-class splits are pooled:
    train = union of all classes' train portions, etc.  Chronological order is
    preserved within each class.

    The resulting three sets do NOT form a single contiguous global timeline,
    and E2 therefore does NOT provide strict global chronological isolation
    across classes.  E2 must not be described as leakage-free or globally
    chronological.

    Parameters
    ----------
    X:
        Feature tensor, shape (n_windows, T, 78), float32.  The global index of
        each window is its position in this array.
    y:
        Multiclass label vector, shape (n_windows,), int16.
    min_train, min_val, min_test:
        Eligibility thresholds for primary per-class reporting.
    """
    n = len(y)

    train_idx: List[np.ndarray] = []
    val_idx: List[np.ndarray] = []
    test_idx: List[np.ndarray] = []

    per_class_meta: Dict[str, object] = {}

    for lbl_str in LABEL_VOCABULARY:
        lbl_code = LABEL_TO_INT.get(lbl_str)
        if lbl_code is None:
            continue

        class_indices = np.where(y == lbl_code)[0]  # already in global order
        nc = len(class_indices)
        if nc == 0:
            per_class_meta[lbl_str] = {
                "n_windows": 0,
                "n_train": 0,
                "n_val": 0,
                "n_test": 0,
                "absent": True,
            }
            continue

        te, ve = _split_indices(nc, train_ratio, val_ratio)
        cls_train = class_indices[:te]
        cls_val   = class_indices[te:ve]
        cls_test  = class_indices[ve:]

        train_idx.append(cls_train)
        val_idx.append(cls_val)
        test_idx.append(cls_test)

        per_class_meta[lbl_str] = {
            "n_windows": nc,
            "n_train": len(cls_train),
            "n_val": len(cls_val),
            "n_test": len(cls_test),
        }

    # Concatenate and sort each pool by global index so the arrays are
    # deterministically ordered (useful for reproducibility checks).
    def _pool(parts: List[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty(0, dtype=np.int64)
        combined = np.concatenate(parts)
        combined.sort()
        return combined

    tr_idx = _pool(train_idx)
    va_idx = _pool(val_idx)
    te_idx = _pool(test_idx)

    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val,   y_val   = X[va_idx], y[va_idx]
    X_test,  y_test  = X[te_idx], y[te_idx]

    present_classes = set(_class_distribution(y).keys())
    test_dist = _class_distribution(y_test)
    rare = [
        cls for cls in LABEL_VOCABULARY
        if cls in present_classes and test_dist.get(cls, 0) < rare_class_min_test_windows
    ]

    eligibility = class_eligibility(
        y_train, y_val, y_test,
        min_train=min_train, min_val=min_val, min_test=min_test,
    )

    metadata: Dict[str, object] = {
        "experiment": "2_per_class_chronological",
        "role": "multiclass separability",
        "primary_experiment": "A",
        "is_primary_benchmark": True,
        "target": "multiclass (15-label vocabulary)",
        "description": (
            "Within each class, windows are sorted by their global index and "
            "split 70/10/20.  The per-class splits are pooled into "
            "train / val / test sets.  Chronological order is preserved within "
            "each class."
        ),
        "interpretation": (
            "E2 is the primary supervised multiclass experiment.  It measures "
            "whether the representation separates the attack classes when every "
            "class is actually represented in training."
        ),
        "globally_chronological": False,
        "leakage_free": False,
        "chronological_scope": "within each class only",
        "temporal_isolation_caveat": (
            "E2 does NOT provide strict global chronological isolation across "
            "classes.  A training window of one class can be later in "
            "wall-clock time than a test window of another class.  E2 must not "
            "be described as leakage-free or globally chronological.  For a "
            "strictly chronological protocol see Experiment 1 (multiclass "
            "diagnostic) and Experiment 3 (binary)."
        ),
        "n_windows_total": n,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "train_class_distribution": _class_distribution(y_train),
        "val_class_distribution": _class_distribution(y_val),
        "test_class_distribution": test_dist,
        "per_class_split_counts": per_class_meta,
        "class_eligibility": eligibility,
        "rare_class_min_test_windows": rare_class_min_test_windows,
        "rare_classes": rare,
        "rare_class_note": (
            "Classes listed here have fewer than rare_class_min_test_windows "
            "test windows.  Retained in all splits; excluded from primary "
            "per-class performance claims.  Never silently merged."
        ),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 10),
    }

    return SplitResult(
        arrays=SplitArrays(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
        ),
        metadata=metadata,
    )


def experiment2_per_class_chronological_indices(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    rare_class_min_test_windows: int = 30,
    min_train: int = ELIGIBILITY_MIN_TRAIN,
    min_val: int = ELIGIBILITY_MIN_VAL,
    min_test: int = ELIGIBILITY_MIN_TEST,
) -> IndexSplitResult:
    """Return E2 membership and metadata without materialising feature tensors.

    The canonical E2 implementation is reused with the global index itself as
    its lightweight ``X`` payload.  This keeps one source of split logic while
    avoiding three fancy-indexed copies of the 4.53-GiB feature tensor.
    """
    marker = np.arange(len(y), dtype=np.int64)
    split = experiment2_per_class_chronological(
        marker,
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        rare_class_min_test_windows=rare_class_min_test_windows,
        min_train=min_train,
        min_val=min_val,
        min_test=min_test,
    )
    return IndexSplitResult(
        indices=IndexSplitArrays(
            train=split.arrays.X_train,
            val=split.arrays.X_val,
            test=split.arrays.X_test,
        ),
        metadata=split.metadata,
    )


# ---------------------------------------------------------------------------
# Experiment 3 — strict chronological binary (PRIMARY EXPERIMENT B)
# ---------------------------------------------------------------------------

def experiment3_binary_chronological(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> SplitResult:
    """Strict global chronological 70 / 10 / 20 split with a binary target.

    PRIMARY EXPERIMENT B.  The window sequence, its ordering and the cut points
    are identical to Experiment 1 — only the target changes:
    ``Benign -> 0``, every non-Benign label ``-> 1``.

    Because the target no longer names attack subclasses, an attack family that
    appears only in the later part of the capture schedule is still a valid test
    case: the model is asked "attack or not", not "which attack".  That makes a
    strict chronological protocol usable as a primary benchmark here, which it
    is not for multiclass (see Experiment 1).

    No shuffling and no reordering: the original row/window ordering from
    windowing is preserved, and the raw data is untouched.

    Parameters
    ----------
    X:
        Feature tensor, shape (n_windows, T, 78), float32.
    y:
        **Multiclass** label vector, shape (n_windows,), int16.  It is converted
        to binary codes here via the canonical mapping, so callers pass the same
        ``y`` they pass to E1 and E2.
    """
    n = len(y)
    y_bin = binary_codes_from_multiclass(y)

    train_end, val_end = _split_indices(n, train_ratio, val_ratio)

    # Basic slicing → views into X, same as E1.
    X_train, y_train = X[:train_end], y_bin[:train_end]
    X_val,   y_val   = X[train_end:val_end], y_bin[train_end:val_end]
    X_test,  y_test  = X[val_end:], y_bin[val_end:]

    # Multiclass composition of each split, kept for diagnostics: it shows
    # which attack families make up the Attack class in each split without
    # changing the target.
    mc_train = _class_distribution(y[:train_end])
    mc_val = _class_distribution(y[train_end:val_end])
    mc_test = _class_distribution(y[val_end:])
    binary_distributions = {
        "train": _binary_distribution(y_train),
        "val": _binary_distribution(y_val),
        "test": _binary_distribution(y_test),
    }
    both_classes_in_all_splits = all(
        dist["Benign"] > 0 and dist["Attack"] > 0
        for dist in binary_distributions.values()
    )

    metadata: Dict[str, object] = {
        "experiment": "3_binary_chronological",
        "role": "strict chronological binary classification",
        "primary_experiment": "B",
        "is_primary_benchmark": True,
        "target": "binary (0 = Benign, 1 = Attack)",
        "description": (
            "Sequential 70/10/20 split on the globally-ordered T=5 window "
            "sequence with a binary target.  Same ordering and same cut points "
            "as Experiment 1; only the label encoding differs."
        ),
        "interpretation": (
            "E3 evaluates temporal generalization without requiring unseen "
            "attack subclasses to be identified. Split-specific binary class "
            "availability is reported explicitly; primary supervised metrics "
            "require both binary classes in the relevant evaluation split."
        ),
        "globally_chronological": True,
        "split_index_overlap": False,
        "leakage_free": False,
        "leakage_caveat": (
            "The E3 window indices are disjoint and strictly ordered. This is "
            "a strict temporal split, not a zero-leakage claim: retained "
            "temporal-duplicate feature states can recur across boundaries."
        ),
        "ordering_note": (
            "Original row and window ordering from the windowing pass is "
            "preserved exactly.  No shuffling, no re-sorting, no modification "
            "of the raw data."
        ),
        "binary_mapping": binary_mapping_dict(),
        "n_windows_total": n,
        "split_boundaries": {
            "train_end_idx": train_end,
            "val_end_idx": val_end,
        },
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "both_binary_classes_present_in_all_splits": both_classes_in_all_splits,
        "train_binary_distribution": binary_distributions["train"],
        "val_binary_distribution": binary_distributions["val"],
        "test_binary_distribution": binary_distributions["test"],
        "train_multiclass_composition": mc_train,
        "val_multiclass_composition": mc_val,
        "test_multiclass_composition": mc_test,
        "multiclass_composition_note": (
            "Recorded for diagnostics only.  The E3 target is binary; the "
            "multiclass mapping is unchanged and remains canonical for E1/E2."
        ),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 10),
    }

    return SplitResult(
        arrays=SplitArrays(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
        ),
        metadata=metadata,
    )
