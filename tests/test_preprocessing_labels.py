"""Tests for src.preprocessing.labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.labels import (
    ABSENT_CLASSES,
    ATTACK_LABELS,
    BENIGN_LABEL,
    BINARY_ATTACK,
    BINARY_BENIGN,
    BINARY_INT_TO_NAME,
    BINARY_LABEL_TO_INT,
    INT_TO_LABEL,
    LABEL_TO_INT,
    LABEL_VOCABULARY,
    MULTICLASS_CODE_TO_BINARY,
    binary_codes_from_multiclass,
    binary_mapping_dict,
    class_mapping_dict,
    decode_labels,
    encode_binary_labels,
    encode_labels,
    to_binary_code,
)


class TestVocabulary:
    def test_exactly_15_labels(self):
        assert len(LABEL_VOCABULARY) == 15

    def test_alphabetical_order(self):
        assert LABEL_VOCABULARY == sorted(LABEL_VOCABULARY)

    def test_no_duplicates(self):
        assert len(set(LABEL_VOCABULARY)) == len(LABEL_VOCABULARY)

    def test_benign_present(self):
        assert "Benign" in LABEL_VOCABULARY

    def test_heartbleed_absent_from_vocabulary(self):
        assert "Heartbleed" not in LABEL_VOCABULARY

    def test_heartbleed_in_absent_classes(self):
        assert "Heartbleed" in ABSENT_CLASSES

    def test_int_to_label_covers_all(self):
        assert set(INT_TO_LABEL.values()) == set(LABEL_VOCABULARY)

    def test_label_to_int_covers_all(self):
        assert set(LABEL_TO_INT.keys()) == set(LABEL_VOCABULARY)

    def test_round_trip(self):
        for i, lbl in enumerate(LABEL_VOCABULARY):
            assert LABEL_TO_INT[lbl] == i
            assert INT_TO_LABEL[i] == lbl

    def test_known_labels_spot_check(self):
        expected = [
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
        assert LABEL_VOCABULARY == expected


class TestEncodeLabels:
    def test_basic_encoding(self):
        s = pd.Series(["Benign", "Bot"])
        codes = encode_labels(s)
        assert codes.dtype == np.int16
        assert codes[0] == LABEL_TO_INT["Benign"]
        assert codes[1] == LABEL_TO_INT["Bot"]

    def test_all_labels_encode(self):
        s = pd.Series(LABEL_VOCABULARY)
        codes = encode_labels(s)
        for i, lbl in enumerate(LABEL_VOCABULARY):
            assert codes[i] == LABEL_TO_INT[lbl]

    def test_unknown_label_raises(self):
        s = pd.Series(["Benign", "Heartbleed"])
        with pytest.raises(ValueError, match="not in the fixed vocabulary"):
            encode_labels(s)

    def test_empty_series(self):
        codes = encode_labels(pd.Series([], dtype=str))
        assert len(codes) == 0


class TestDecodeLabels:
    def test_round_trip(self):
        original = ["Benign", "SQL Injection", "Bot"]
        codes = encode_labels(pd.Series(original))
        decoded = decode_labels(codes)
        assert decoded == original

    def test_single(self):
        assert decode_labels(np.array([0], dtype=np.int16)) == [LABEL_VOCABULARY[0]]


class TestClassMappingDict:
    def test_keys_present(self):
        d = class_mapping_dict()
        assert "n_classes" in d
        assert "int_to_label" in d
        assert "label_to_int" in d
        assert "absent_from_corpus" in d

    def test_n_classes(self):
        assert class_mapping_dict()["n_classes"] == 15

    def test_heartbleed_absent_annotation(self):
        d = class_mapping_dict()
        assert "Heartbleed" in d["absent_from_corpus"]
        assert "absent from corpus" in d["absent_from_corpus"]["Heartbleed"]


# ---------------------------------------------------------------------------
# §6 — canonical binary target
# ---------------------------------------------------------------------------

class TestBinaryMapping:
    """Requirements: every original label maps to exactly one binary class;
    Benign → 0; every attack → 1; no label silently dropped."""

    def test_codes_are_zero_and_one(self):
        assert BINARY_BENIGN == 0
        assert BINARY_ATTACK == 1
        assert BINARY_INT_TO_NAME == {0: "Benign", 1: "Attack"}

    def test_every_label_present_exactly_once(self):
        assert set(BINARY_LABEL_TO_INT) == set(LABEL_VOCABULARY)
        assert len(BINARY_LABEL_TO_INT) == len(LABEL_VOCABULARY) == 15

    def test_no_label_dropped(self):
        for lbl in LABEL_VOCABULARY:
            assert lbl in BINARY_LABEL_TO_INT

    def test_each_label_maps_to_exactly_one_class(self):
        for lbl in LABEL_VOCABULARY:
            code = BINARY_LABEL_TO_INT[lbl]
            assert code in (BINARY_BENIGN, BINARY_ATTACK)
            assert isinstance(code, int)

    def test_benign_maps_to_zero(self):
        assert BINARY_LABEL_TO_INT[BENIGN_LABEL] == BINARY_BENIGN
        assert BINARY_LABEL_TO_INT["Benign"] == 0
        assert to_binary_code("Benign") == 0

    def test_all_fourteen_attacks_map_to_one(self):
        assert len(ATTACK_LABELS) == 14
        assert BENIGN_LABEL not in ATTACK_LABELS
        for lbl in ATTACK_LABELS:
            assert BINARY_LABEL_TO_INT[lbl] == BINARY_ATTACK
            assert to_binary_code(lbl) == 1

    def test_partition_counts(self):
        codes = list(BINARY_LABEL_TO_INT.values())
        assert codes.count(BINARY_BENIGN) == 1
        assert codes.count(BINARY_ATTACK) == 14

    def test_unknown_label_raises_rather_than_defaulting_to_attack(self):
        with pytest.raises(ValueError, match="refusing to guess"):
            to_binary_code("Heartbleed")

    def test_multiclass_mapping_unchanged(self):
        """E1/E2 multiclass encoding must not be perturbed by the binary work."""
        assert len(LABEL_TO_INT) == 15
        for i, lbl in enumerate(LABEL_VOCABULARY):
            assert LABEL_TO_INT[lbl] == i
            assert INT_TO_LABEL[i] == lbl

    def test_code_mapping_covers_every_multiclass_code(self):
        assert set(MULTICLASS_CODE_TO_BINARY) == set(range(len(LABEL_VOCABULARY)))
        for code, bcode in MULTICLASS_CODE_TO_BINARY.items():
            expected = BINARY_LABEL_TO_INT[INT_TO_LABEL[code]]
            assert bcode == expected


class TestEncodeBinaryLabels:
    def test_basic(self):
        s = pd.Series(["Benign", "Bot", "SQL Injection"])
        codes = encode_binary_labels(s)
        assert codes.dtype == np.int16
        assert codes.tolist() == [0, 1, 1]

    def test_all_labels(self):
        codes = encode_binary_labels(pd.Series(LABEL_VOCABULARY))
        assert codes.tolist() == [
            0 if lbl == "Benign" else 1 for lbl in LABEL_VOCABULARY
        ]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="not in the fixed vocabulary"):
            encode_binary_labels(pd.Series(["Benign", "Heartbleed"]))

    def test_empty(self):
        assert len(encode_binary_labels(pd.Series([], dtype=str))) == 0


class TestBinaryCodesFromMulticlass:
    def test_matches_string_path(self):
        """Deriving from codes agrees with deriving from label strings."""
        s = pd.Series(LABEL_VOCABULARY)
        via_strings = encode_binary_labels(s)
        via_codes = binary_codes_from_multiclass(encode_labels(s))
        np.testing.assert_array_equal(via_strings, via_codes)

    def test_dtype_and_shape(self):
        codes = np.array([0, 1, 2, 14], dtype=np.int16)
        out = binary_codes_from_multiclass(codes)
        assert out.dtype == np.int16
        assert out.shape == codes.shape
        assert out.tolist() == [0, 1, 1, 1]

    def test_out_of_vocabulary_code_raises(self):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            binary_codes_from_multiclass(np.array([0, 99], dtype=np.int16))

    def test_empty(self):
        out = binary_codes_from_multiclass(np.array([], dtype=np.int16))
        assert len(out) == 0


class TestBinaryMappingDict:
    def test_keys(self):
        d = binary_mapping_dict()
        for key in (
            "n_classes", "int_to_name", "label_to_binary", "benign_label",
            "n_attack_labels", "attack_labels", "policy",
        ):
            assert key in d

    def test_values(self):
        d = binary_mapping_dict()
        assert d["n_classes"] == 2
        assert d["benign_label"] == "Benign"
        assert d["n_attack_labels"] == 14
        assert len(d["label_to_binary"]) == 15

    def test_policy_states_multiclass_unchanged(self):
        assert "multiclass mapping is unchanged" in binary_mapping_dict()["policy"]
