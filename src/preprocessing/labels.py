"""Fixed 15-label vocabulary for the CIC-IDS2018 corpus.

The vocabulary is the exact set of 15 label strings measured in Stage B
(see ``results/data_profile/global_dataset_summary.json``).

Heartbleed is absent from this corpus.  It is documented here explicitly so
that any downstream code that evaluates per-class metrics can mark it as
"not evaluated — absent from corpus" rather than silently omitting it.

Two target encodings are defined here and both are canonical:

``LABEL_TO_INT`` / ``INT_TO_LABEL``
    The 15-way multiclass encoding used by Experiment 1 (strict chronological
    multiclass diagnostic) and Experiment 2 (primary multiclass separability).
    This mapping is unchanged.

``BINARY_LABEL_TO_INT`` / ``BINARY_INT_TO_NAME``
    The 2-way encoding used by Experiment 3 (strict chronological binary
    classification): ``Benign -> 0``, every one of the 14 attack labels
    ``-> 1``.  Every label in :data:`LABEL_VOCABULARY` is present in the binary
    mapping; no label is dropped.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical vocabulary
# ---------------------------------------------------------------------------

#: The 15 label strings, in deterministic alphabetical order.
#: These are the EXACT strings as they appear in the CSV files; do not alter
#: capitalisation, spacing or punctuation.
LABEL_VOCABULARY: List[str] = [
    "Benign",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infilteration",
    "SQL Injection",
    "SSH-Bruteforce",
]

#: Classes known to be absent from this corpus.  Kept here so evaluation code
#: can emit an explicit "not evaluated — absent from corpus" entry rather than
#: silently omitting the class.
ABSENT_CLASSES: List[str] = ["Heartbleed"]

#: int → label string
INT_TO_LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABEL_VOCABULARY)}

#: label string → int
LABEL_TO_INT: Dict[str, int] = {label: i for i, label in enumerate(LABEL_VOCABULARY)}


# ---------------------------------------------------------------------------
# Binary target (Experiment 3)
# ---------------------------------------------------------------------------
#: The single label that counts as non-attack traffic.
BENIGN_LABEL: str = "Benign"

#: Binary codes.  These are the canonical E3 targets.
BINARY_BENIGN: int = 0
BINARY_ATTACK: int = 1

#: binary code → human-readable name.
BINARY_INT_TO_NAME: Dict[int, str] = {BINARY_BENIGN: "Benign", BINARY_ATTACK: "Attack"}

#: The 14 attack labels (everything in the vocabulary that is not Benign).
ATTACK_LABELS: List[str] = [lbl for lbl in LABEL_VOCABULARY if lbl != BENIGN_LABEL]

#: label string → binary code.  Built from the full vocabulary so that every
#: label maps to exactly one binary class and none can be silently dropped.
BINARY_LABEL_TO_INT: Dict[str, int] = {
    label: (BINARY_BENIGN if label == BENIGN_LABEL else BINARY_ATTACK)
    for label in LABEL_VOCABULARY
}

#: multiclass code → binary code.  Same mapping expressed over integer codes,
#: so an already-encoded multiclass array can be converted without going back
#: through strings.
MULTICLASS_CODE_TO_BINARY: Dict[int, int] = {
    LABEL_TO_INT[label]: code for label, code in BINARY_LABEL_TO_INT.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_labels(labels: "pd.Series") -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
    """Map a Series of raw label strings to integer codes (int16).

    Raises
    ------
    ValueError
        If any string in *labels* is not in :data:`LABEL_VOCABULARY`.
    """
    import numpy as np

    unknown = set(labels.unique()) - set(LABEL_VOCABULARY)
    if unknown:
        raise ValueError(
            f"Labels not in the fixed vocabulary: {sorted(unknown)}"
        )
    return labels.map(LABEL_TO_INT).to_numpy(dtype=np.int16)


def decode_labels(codes: "np.ndarray") -> List[str]:  # type: ignore[name-defined]  # noqa: F821
    """Map an array of integer codes back to label strings."""
    return [INT_TO_LABEL[int(c)] for c in codes]


def class_mapping_dict() -> Dict[str, object]:
    """Return a JSON-serialisable dict describing the full class mapping.

    Suitable for writing to ``class_mapping.json`` under the artefact directory.
    """
    return {
        "n_classes": len(LABEL_VOCABULARY),
        "int_to_label": {str(k): v for k, v in INT_TO_LABEL.items()},
        "label_to_int": LABEL_TO_INT,
        "absent_from_corpus": {
            cls: "not evaluated — absent from corpus"
            for cls in ABSENT_CLASSES
        },
    }


# ---------------------------------------------------------------------------
# Binary helpers (Experiment 3)
# ---------------------------------------------------------------------------

def to_binary_code(label: str) -> int:
    """Map one raw label string to its binary code (0 = Benign, 1 = Attack).

    Raises
    ------
    ValueError
        If *label* is not in :data:`LABEL_VOCABULARY`.  Unknown labels are
        rejected rather than defaulted, so a vocabulary drift cannot silently
        become "Attack".
    """
    try:
        return BINARY_LABEL_TO_INT[label]
    except KeyError:
        raise ValueError(
            f"Label {label!r} is not in the fixed vocabulary; refusing to guess "
            "a binary class for it."
        ) from None


def encode_binary_labels(labels: "pd.Series") -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
    """Map a Series of raw label strings to binary codes (int16)."""
    import numpy as np

    unknown = set(labels.unique()) - set(BINARY_LABEL_TO_INT)
    if unknown:
        raise ValueError(f"Labels not in the fixed vocabulary: {sorted(unknown)}")
    return labels.map(BINARY_LABEL_TO_INT).to_numpy(dtype=np.int16)


def binary_codes_from_multiclass(codes: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]  # noqa: F821
    """Convert an int multiclass code array to binary codes (int16).

    This is the path used by the pipeline: windowing already produced int16
    multiclass codes, so E3 is derived from those rather than re-reading the
    label strings.  Equivalent to mapping each code through
    :data:`MULTICLASS_CODE_TO_BINARY`.
    """
    import numpy as np

    codes_arr = np.asarray(codes)
    unknown = set(np.unique(codes_arr).tolist()) - set(MULTICLASS_CODE_TO_BINARY)
    if unknown:
        raise ValueError(f"Multiclass codes outside the vocabulary: {sorted(unknown)}")

    # Table lookup: index i holds the binary code for multiclass code i.
    table = np.empty(len(LABEL_VOCABULARY), dtype=np.int16)
    for mc_code, bin_code in MULTICLASS_CODE_TO_BINARY.items():
        table[mc_code] = bin_code
    return table[codes_arr.astype(np.intp)]


def binary_mapping_dict() -> Dict[str, object]:
    """Return a JSON-serialisable dict describing the binary (E3) mapping."""
    return {
        "n_classes": 2,
        "int_to_name": {str(k): v for k, v in BINARY_INT_TO_NAME.items()},
        "label_to_binary": BINARY_LABEL_TO_INT,
        "benign_label": BENIGN_LABEL,
        "n_attack_labels": len(ATTACK_LABELS),
        "attack_labels": ATTACK_LABELS,
        "policy": (
            "Benign -> 0; all 14 non-Benign labels -> 1.  Every label in the "
            "15-label vocabulary appears in this mapping exactly once.  The "
            "multiclass mapping is unchanged and remains canonical for E1/E2."
        ),
    }
