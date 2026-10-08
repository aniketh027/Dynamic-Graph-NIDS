"""Strict bounded raw-feature CSV parsing and authenticated train-state reuse."""
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import BinaryIO

import numpy as np

from backend.app.core.errors import ApplicationError
from src.graph.schema import TEMPORAL_SEMANTICS
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS, SENTINEL_NEGATIVE_COLUMNS
from src.preprocessing.transforms import FitTransformState, apply_transforms

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 25_000
BATCH_SIZE = 256
WINDOW = TEMPORAL_SEMANTICS["sequence_length"]
FLOAT32_MAX = float(np.finfo(np.float32).max)
TRANSFORM_PATH = "data/processed/preprocessed/exp2/scaler_state.json"
TRANSFORM_ID = "exp2/scaler_state.json"
TRANSFORM_SHA256 = "9d9fc971b6a2f2d4bda3b31805a08742a26037a436f2c8130483769b800f7fe2"
# Explicit decimal/IEEE grammar: no arbitrary text, blanks, NA aliases or formulas.
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)", re.IGNORECASE)


@dataclass
class ParsedCSV:
    rows: np.ndarray
    columns_reordered: bool

    @property
    def window_count(self):
        return len(self.rows) // WINDOW


def load_transform(root: Path) -> FitTransformState:
    """Read only this compact state, authenticate bytes, then use frozen APIs."""
    try:
        with (root / TRANSFORM_PATH).open("rb") as source:
            payload = source.read(128 * 1024 + 1)
        if len(payload) > 128 * 1024 or hashlib.sha256(payload).hexdigest() != TRANSFORM_SHA256:
            raise ValueError("state hash")
        raw = json.loads(payload)
        required = {"feature_names", "center", "scale", "train_median", "train_max_finite",
                    "train_min_finite", "sentinel_columns", "sentinel_policy", "scaler", "fit_provenance"}
        if set(raw) != required or raw["feature_names"] != list(MODEL_FEATURE_COLUMNS):
            raise ValueError("state schema")
        if raw["sentinel_columns"] != list(SENTINEL_NEGATIVE_COLUMNS):
            raise ValueError("sentinel schema")
        provenance = raw["fit_provenance"]
        if (provenance["fit_experiment"] != "E2" or provenance["fit_split"] != "train"
                or provenance["applies_to_experiments"] != ["E2"]):
            raise ValueError("fit provenance")
        state = FitTransformState.from_dict(raw)
        if any(np.asarray(getattr(state, name)).shape != (78,)
               or not np.isfinite(getattr(state, name)).all()
               for name in ("center", "scale", "train_median", "train_max_finite", "train_min_finite")):
            raise ValueError("state arrays")
        if np.any(state.scale <= 0) or state.to_dict() != {k: v for k, v in raw.items() if k != "fit_provenance"}:
            raise ValueError("state semantics")
        return state
    except Exception as exc:
        raise ApplicationError("TRANSFORM_STATE_INVALID", "Frozen E2 transform authentication failed.", 503) from exc


def parse_csv(source: BinaryIO) -> ParsedCSV:
    """Source is already byte-bounded. csv.reader sees original, unmangled headers."""
    source.seek(0)
    text = io.TextIOWrapper(source, encoding="utf-8-sig", newline="", errors="strict")
    try:
        reader = csv.reader(text, strict=True)
        header = next(reader, None)
        if not header:
            raise ApplicationError("EMPTY_FILE", "Supply a CSV header and at least one flow record.", 422)
        if len(set(header)) != len(header):
            raise ApplicationError("DUPLICATE_COLUMN", "CSV header names must be unique.", 422)
        expected = set(MODEL_FEATURE_COLUMNS)
        if set(header) - expected:
            raise ApplicationError("UNEXPECTED_COLUMN", "Only the 78 model features are accepted; remove Timestamp, Label and other metadata.", 422)
        if expected - set(header):
            raise ApplicationError("MISSING_COLUMN", "Include every canonical model feature from /api/v1/schema.", 422)
        order = [header.index(name) for name in MODEL_FEATURE_COLUMNS]
        rows = np.empty((MAX_ROWS, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32)
        count = 0
        for fields in reader:
            if count == MAX_ROWS:
                raise ApplicationError("TOO_MANY_ROWS", "At most 25,000 flow records are accepted.", 413)
            if len(fields) != len(header):
                raise ApplicationError("CSV_PARSE_ERROR", "Every CSV row must contain exactly 78 fields.", 422)
            for j, source_index in enumerate(order):
                value = fields[source_index].strip()
                if NUMBER.fullmatch(value) is None:
                    raise ApplicationError("NON_NUMERIC_FEATURE", f"Flow index {count}, feature {j} requires a numeric value (NaN and +/-inf are allowed).", 422)
                number = float(value)
                # Overflowing decimal text must not masquerade as explicit IEEE infinity.
                explicit_nonfinite = value.lower().lstrip("+-") in {"nan", "inf", "infinity"}
                if not explicit_nonfinite and (not np.isfinite(number) or abs(number) > FLOAT32_MAX):
                    raise ApplicationError("NON_NUMERIC_FEATURE", "Finite values must fit float32; use explicit NaN or +/-inf for non-finite values.", 422)
                rows[count, j] = number
            count += 1
        if count == 0:
            raise ApplicationError("EMPTY_FILE", "The CSV contains no flow records.", 422)
        return ParsedCSV(rows[:count], tuple(header) != MODEL_FEATURE_COLUMNS)
    except (csv.Error, UnicodeError) as exc:
        raise ApplicationError("CSV_PARSE_ERROR", "Use a well-formed, uncompressed UTF-8 CSV with valid quoting.", 422) from exc
    finally:
        text.detach()  # The request owns and closes the underlying spool.


def transform_batch(raw: np.ndarray, state: FitTransformState) -> np.ndarray:
    try:
        output = apply_transforms(raw, state)
        if output.shape != raw.shape or output.dtype != np.float32 or not np.isfinite(output).all():
            raise ValueError("invalid transformed windows")
        return output
    except Exception as exc:
        raise ApplicationError("TRANSFORM_FAILED", "Frozen preprocessing could not produce finite float32 windows.", 422) from exc
