"""Canonical duplicate classification for the CIC-IDS2018 corpus.

This module implements ONE duplicate-classification pass.  Every duplicate
number reported anywhere in Stage C comes from this pass, so the counts cannot
drift between modules.

The cascade is strictly mutually exclusive and is applied BEFORE windowing.
Each usable row ends in exactly one of three final states:

``CONFLICTING``
    Excluded by Level 1.

``EXACT_DUPLICATE``
    Excluded by Level 2.

``RETAINED_NON_DUPLICATE``
    Retained after Levels 1 and 2.  Level 3 adds a temporal-duplicate indicator
    to repeated retained observations; it does not change this final state.

For reporting, the retained state is further partitioned into two mutually
exclusive subcategories:

LEVEL 1 — ``CONFLICTING``  (excluded)
    Group rows by the 78 model features alone.  If a feature-vector group
    carries more than one distinct ``Label``, every row in that group is
    conflicting.  These rows are **ambiguous under the feature representation**:
    the 78 features do not determine the label.  This is a statement about the
    representation, not proof that the rows are noise.

LEVEL 2 — ``EXACT_DUPLICATE``  (excluded after the first occurrence)
    Considered only for rows NOT already flagged conflicting.  Group by
    (78 features + Timestamp + Label).  The first occurrence in deterministic
    source order is retained; every later occurrence is an exact duplicate and
    is excluded.

LEVEL 3 — ``TEMPORAL_DUPLICATE``  (retained)
    Considered only for rows that survived Levels 1 and 2.  Group by
    (78 features + Label).  A repeat of a feature/label state that already
    appeared at a different Timestamp / source position is a temporal
    duplicate.  Temporal duplicates are RETAINED — they are separate flow
    observations, not noise.

``UNIQUE``  (retained)
    Everything else: the first time a given (features, label) state is seen.

Precedence is total: Level 1 is decided before Level 2, and Level 2 before
Level 3, so no row can hold two exclusion states.  The temporal flag applies
only to retained rows.

Accounting identity
-------------------
The four categories partition the usable rows exactly::

    usable_rows = conflicting_rows
                + exact_duplicate_rows_removed
                + retained_rows

    retained_rows = temporal_duplicate_rows_retained + unique_rows
    clean_rows  = retained_rows
                = usable_rows - conflicting_rows - exact_duplicate_rows_removed

Note that temporal duplicates are a SUBSET of the clean rows, not a further
subtraction from them.  An earlier implementation computed
``temporal = total - exact - conflicting - clean``, which is an accounting
identity that returns 0 for any input; that is why temporal duplicates were
previously reported as zero.

Determinism
-----------
"First occurrence" means first in the global capture-order proxy:
``(source file capture date, Original_Index)``.  ``Original_Index`` is the
synthetic 0-based source-row position assigned BEFORE embedded-header
filtering, so it is stable regardless of how many rows are later removed.
Files are visited in the capture-date order given by *csv_paths* and chunks in
file order, so the first row encountered for a group is exactly the earliest in
that global order.

Memory and hashing strategy
---------------------------
Holding the full 16.2M x 78 float64 matrix would need ~10 GB, so the corpus is
streamed in bounded chunks.  Deterministic 64-bit fingerprints select candidate
buckets, but they are never accepted as proof of equality.  Level-1 feature
payloads are kept in a temporary disk-backed fixed-record store and compared
exactly inside matching buckets.  Levels 2 and 3 compare exact canonical keys
derived from those verified feature groups.  Only hash buckets that repeat need
retain Level-2/Level-3 candidate payloads in memory.

The raw CSVs under ``data/raw/`` are never modified.
"""

from __future__ import annotations

import mmap
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.labels import LABEL_TO_INT
from src.preprocessing.schema import (
    EMBEDDED_HEADER_LABEL,
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    ORIGINAL_INDEX_COLUMN,
    TIMESTAMP_COLUMN,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UINT64_MAX = np.uint64(0xFFFF_FFFF_FFFF_FFFF)
_FNV_OFFSET = np.uint64(0xCBF2_9CE4_8422_2325)
_FNV_PRIME = np.uint64(0x0000_0100_0000_01B3)
_HASH_MIX = np.uint64(0x9E37_79B9_7F4A_7C15)

#: Final duplicate categories.  Exactly one applies to each usable row.
CATEGORY_CONFLICTING = "CONFLICTING"
CATEGORY_EXACT_DUPLICATE = "EXACT_DUPLICATE"
CATEGORY_TEMPORAL_DUPLICATE = "TEMPORAL_DUPLICATE"
CATEGORY_UNIQUE = "UNIQUE"

#: Categories whose rows are excluded from the clean experimental dataset.
EXCLUDED_CATEGORIES = (CATEGORY_CONFLICTING, CATEGORY_EXACT_DUPLICATE)
#: Categories whose rows are retained.
RETAINED_CATEGORIES = (CATEGORY_TEMPORAL_DUPLICATE, CATEGORY_UNIQUE)


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _fnv1a_continue(h: int, data: bytes) -> int:
    """Fold *data* into an in-progress 64-bit FNV-1a state *h*."""
    prime = int(_FNV_PRIME)
    mask = int(_UINT64_MAX)
    for byte in data:
        h ^= byte
        h = (h * prime) & mask
    return h


def _fnv1a_uint64(data: bytes) -> int:
    """Compute a 64-bit FNV-1a hash of *data*."""
    return _fnv1a_continue(int(_FNV_OFFSET), data)


def _row_hashes(
    features: np.ndarray,     # shape (N, 78), float64
    ts_bytes: np.ndarray,     # shape (N,), object (raw Timestamp string bytes)
    label_codes: np.ndarray,  # shape (N,), int16
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (feat_hash, feat_ts_label_hash) arrays, both uint64, shape (N,).

    Compatibility helper used by hashing tests.  The second result is the
    exact-row fingerprint (features + label + timestamp).
    """
    feat_hash, _feat_label_hash, exact_hash = _cascade_hashes(
        features, ts_bytes, label_codes
    )
    return feat_hash, exact_hash


def _combine_hashes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Order-dependent vectorized uint64 hash combine."""
    with np.errstate(over="ignore"):
        return np.bitwise_xor(
            left,
            np.multiply(right, _HASH_MIX, dtype=np.uint64),
        )


def _cascade_hashes(
    features: np.ndarray,
    timestamps: np.ndarray,
    label_codes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectorized Level-1, Level-3 and Level-2 fingerprints.

    The tuple is ``(feature, feature_label, exact)``.  Equality semantics match
    the cascade keys; the combine order itself is irrelevant as long as it is
    deterministic and different key components cannot be silently omitted.
    """
    feature_frame = pd.DataFrame(features, copy=False)
    feat_hash = pd.util.hash_pandas_object(
        feature_frame, index=False
    ).to_numpy(dtype=np.uint64, copy=False)
    label_hash = pd.util.hash_array(
        np.asarray(label_codes, dtype=np.int16)
    ).astype(np.uint64, copy=False)
    timestamp_hash = pd.util.hash_array(
        np.asarray(timestamps, dtype=object)
    ).astype(np.uint64, copy=False)
    feat_label_hash = _combine_hashes(feat_hash, label_hash)
    exact_hash = _combine_hashes(feat_label_hash, timestamp_hash)
    return feat_hash, feat_label_hash, exact_hash


# ---------------------------------------------------------------------------
# Per-file chunk loader
# ---------------------------------------------------------------------------

@dataclass
class _ChunkCounts:
    """Row counts for one raw chunk, before any exclusion."""

    raw: int = 0
    embedded_header: int = 0
    unknown_label: int = 0


def _iter_chunks(
    csv_path: Path,
    chunk_size: int,
    row_limit: Optional[int],
) -> Iterator[Tuple[pd.DataFrame, _ChunkCounts]]:
    """Yield ``(chunk, counts)`` for *csv_path*.

    The chunk holds a synthetic ``Original_Index`` (0-based source-row position
    counted over ALL raw rows, including embedded header rows, assigned before
    any filtering), the 78 model features as float64, ``Timestamp`` and
    ``Label``.

    Embedded header rows and rows whose label is outside the fixed vocabulary
    are removed from the yielded chunk, but are counted in *counts* so the
    accounting can reconcile against the raw row total.
    """
    usecols = list(MODEL_FEATURE_COLUMNS) + [TIMESTAMP_COLUMN, LABEL_COLUMN]
    rows_read = 0
    row_offset = 0  # counts every raw row, including removed ones
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=str,          # embedded header rows would break a float cast
        chunksize=chunk_size,
        low_memory=False,
    ):
        if row_limit is not None:
            remaining = row_limit - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

        n = len(chunk)
        counts = _ChunkCounts(raw=n)

        chunk = chunk.copy()
        # Assign the source position BEFORE any row is dropped.
        chunk[ORIGINAL_INDEX_COLUMN] = np.arange(
            row_offset, row_offset + n, dtype=np.int64
        )

        header_mask = chunk[LABEL_COLUMN] == EMBEDDED_HEADER_LABEL
        counts.embedded_header = int(header_mask.sum())
        if counts.embedded_header:
            chunk = chunk[~header_mask]

        valid_mask = chunk[LABEL_COLUMN].isin(LABEL_TO_INT)
        counts.unknown_label = int((~valid_mask).sum())
        if counts.unknown_label:
            chunk = chunk[valid_mask]

        for col in MODEL_FEATURE_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        yield chunk, counts

        rows_read += n
        row_offset += n
        if row_limit is not None and rows_read >= row_limit:
            break


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

class AccountingError(RuntimeError):
    """Raised when the duplicate accounting fails to reconcile."""


@dataclass
class CleaningStats:
    """Canonical duplicate accounting for the whole corpus.

    Every field is measured by a single pass in :func:`build_keep_index` and is
    computed BEFORE windowing.
    """

    #: Every data row read from the CSVs, including embedded header rows.
    raw_rows: int = 0
    #: Repeated header lines concatenated into the data (Label == "Label").
    embedded_header_rows: int = 0
    #: Rows whose Label is outside the fixed 15-label vocabulary.
    unknown_label_rows: int = 0
    #: raw_rows - embedded_header_rows - unknown_label_rows.
    usable_rows: int = 0

    #: LEVEL 1 — rows in feature-vector groups carrying >1 distinct label.
    conflicting_rows: int = 0
    #: Number of distinct feature-vector groups that are conflicting.
    conflicting_groups: int = 0
    #: LEVEL 2 — non-first occurrences of (features + Timestamp + Label).
    exact_duplicate_rows_removed: int = 0
    #: LEVEL 3 — repeats of a (features + Label) state; RETAINED.
    temporal_duplicate_rows_retained: int = 0
    #: First occurrence of each (features + Label) state; RETAINED.
    unique_rows: int = 0
    #: Rows entering windowing = temporal_duplicate_rows_retained + unique_rows.
    clean_rows: int = 0

    #: Source files processed, in capture-date order.
    source_files: List[str] = field(default_factory=list)

    @property
    def retained_rows(self) -> int:
        """Rows in final state ``RETAINED_NON_DUPLICATE``.

        Temporal duplication is an indicator within this state, not an
        additional exclusion or a fourth top-level final state.
        """
        return self.clean_rows

    # -- reconciliation ------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`AccountingError` if the counts do not reconcile.

        Checks the identities that make the cascade mutually exclusive: the
        three final states partition the usable rows, and the retained state
        is exactly the temporal-indicator/unique breakdown.
        """
        expected_usable = (
            self.raw_rows - self.embedded_header_rows - self.unknown_label_rows
        )
        if self.usable_rows != expected_usable:
            raise AccountingError(
                f"usable_rows={self.usable_rows:,} but raw_rows - headers - "
                f"unknown = {expected_usable:,}"
            )

        final_state_partition = (
            self.conflicting_rows
            + self.exact_duplicate_rows_removed
            + self.retained_rows
        )
        if final_state_partition != self.usable_rows:
            raise AccountingError(
                "final row states do not partition the usable rows: "
                f"{final_state_partition:,} categorised vs "
                f"{self.usable_rows:,} usable"
            )

        expected_clean = self.temporal_duplicate_rows_retained + self.unique_rows
        if self.clean_rows != expected_clean:
            raise AccountingError(
                f"clean_rows={self.clean_rows:,} but temporal + unique = "
                f"{expected_clean:,}"
            )

        alt_clean = (
            self.usable_rows
            - self.conflicting_rows
            - self.exact_duplicate_rows_removed
        )
        if self.clean_rows != alt_clean:
            raise AccountingError(
                f"clean_rows={self.clean_rows:,} but usable - conflicting - "
                f"exact = {alt_clean:,}"
            )

    def accounting_table(self) -> List[Dict[str, object]]:
        """Return the accounting as an ordered list of rows for reporting."""
        return [
            {"quantity": "raw rows", "count": self.raw_rows, "disposition": "read"},
            {
                "quantity": "embedded header rows",
                "count": self.embedded_header_rows,
                "disposition": "removed (not data)",
            },
            {
                "quantity": "unknown label rows",
                "count": self.unknown_label_rows,
                "disposition": "removed (outside vocabulary)",
            },
            {"quantity": "usable rows", "count": self.usable_rows, "disposition": "classified below"},
            {
                "quantity": "conflicting rows",
                "count": self.conflicting_rows,
                "disposition": "excluded (level 1)",
            },
            {
                "quantity": "exact duplicate rows removed",
                "count": self.exact_duplicate_rows_removed,
                "disposition": "excluded (level 2)",
            },
            {
                "quantity": "retained rows",
                "count": self.retained_rows,
                "disposition": "retained after levels 1 and 2",
            },
            {
                "quantity": "temporal duplicate rows among retained data",
                "count": self.temporal_duplicate_rows_retained,
                "disposition": "retained; level-3 indicator subset",
            },
            {
                "quantity": "unique rows",
                "count": self.unique_rows,
                "disposition": "retained",
            },
            {
                "quantity": "final clean rows",
                "count": self.clean_rows,
                "disposition": "enter windowing",
            },
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "raw_rows": self.raw_rows,
            "embedded_header_rows": self.embedded_header_rows,
            "unknown_label_rows": self.unknown_label_rows,
            "usable_rows": self.usable_rows,
            "conflicting_rows": self.conflicting_rows,
            "conflicting_groups": self.conflicting_groups,
            "exact_duplicate_rows_removed": self.exact_duplicate_rows_removed,
            "temporal_duplicate_rows_retained": self.temporal_duplicate_rows_retained,
            "unique_rows": self.unique_rows,
            "retained_rows": self.retained_rows,
            "clean_rows": self.clean_rows,
            "source_files": self.source_files,
            "accounting_table": self.accounting_table(),
            "identities": {
                "final_state_partition": (
                    "usable_rows = conflicting + exact_removed + retained_rows"
                ),
                "retained_breakdown": (
                    "retained_rows = temporal_retained + unique"
                ),
                "clean": (
                    "clean_rows = retained_rows = temporal_retained + unique = "
                    "usable_rows - conflicting - exact_removed"
                ),
                "note": (
                    "temporal duplicates are a SUBSET of clean_rows, not a "
                    "further subtraction from them"
                ),
            },
            "policy": {
                "precedence": "level 1 conflicting -> level 2 exact -> level 3 temporal",
                "mutually_exclusive": True,
                "computed_before_windowing": True,
                "final_states": [
                    "CONFLICTING",
                    "EXACT_DUPLICATE",
                    "RETAINED_NON_DUPLICATE",
                ],
                "temporal_is_retained_indicator": True,
                "level_1_conflicting": (
                    "same 78-feature vector carries >1 distinct Label — excluded; "
                    "ambiguous under the feature representation, NOT proven noise"
                ),
                "level_2_exact": (
                    "same (78 features + Timestamp + Label); first source "
                    "occurrence retained, later occurrences excluded"
                ),
                "level_3_temporal": (
                    "same (78 features + Label) recurring at a different "
                    "Timestamp / source position — RETAINED as a distinct flow "
                    "observation, NOT treated as noise"
                ),
                "first_occurrence_order": (
                    "(source file capture date, Original_Index); Original_Index "
                    "is assigned before embedded-header filtering"
                ),
                "raw_files_modified": False,
                "hashing": (
                    "deterministic vectorized pandas/NumPy uint64 fingerprints "
                    "select candidate buckets only; exact canonical payload "
                    "comparison inside every matching bucket determines equality"
                ),
            },
        }


@dataclass
class CleaningResult:
    """Output of :func:`build_keep_index`."""

    #: Sequence of ``(source_file_stem, original_index_array)`` in capture-date
    #: order.  Only retained rows are present.
    kept_per_file: List[Tuple[str, np.ndarray]]
    stats: CleaningStats


# ---------------------------------------------------------------------------
# Collision-safe exact feature storage and bucket helpers
# ---------------------------------------------------------------------------

_FEATURE_STATUS_BITS = 4
_FEATURE_STATUS_MASK = (1 << _FEATURE_STATUS_BITS) - 1
_FEATURE_CONFLICTING = _FEATURE_STATUS_MASK


def _pack_feature_entry(record_id: int, label_status: int) -> int:
    """Pack a disk record id and first-label/conflict status into one int."""
    return (int(record_id) << _FEATURE_STATUS_BITS) | int(label_status)


def _unpack_feature_entry(entry: int) -> Tuple[int, int]:
    """Return ``(record_id, label_status)`` from a packed feature entry."""
    return int(entry) >> _FEATURE_STATUS_BITS, int(entry) & _FEATURE_STATUS_MASK


def _canonical_timestamp(value: object) -> Optional[str]:
    """Return the exact timestamp key used for payload equality."""
    return None if pd.isna(value) else str(value)


class _FeaturePayloadStore:
    """Temporary disk-backed fixed-record store for exact feature comparison.

    Only one copy of each exact Level-1 feature vector is written.  The file is
    grown geometrically in bounded steps and memory-mapped, so the full corpus
    feature matrix is never materialised in RAM.  Byte equality is the fast
    path; numerical equality with NaNs equal preserves pandas-style canonical
    value semantics for signed zero and alternate NaN payloads.
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = int(n_features)
        self.record_bytes = self.n_features * np.dtype(np.float64).itemsize
        handle = tempfile.NamedTemporaryFile(
            prefix="stage_c_feature_payloads_", suffix=".bin", delete=False
        )
        self.path = Path(handle.name)
        handle.close()
        self._file = self.path.open("r+b")
        self._mapping: Optional[mmap.mmap] = None
        self._view: Optional[memoryview] = None
        self.capacity = 0
        self.count = 0

    def _ensure_capacity(self) -> None:
        if self.count < self.capacity:
            return
        if self.capacity == 0:
            new_capacity = 4096
        elif self.capacity < 1_048_576:
            new_capacity = self.capacity * 2
        else:
            new_capacity = self.capacity + 1_048_576

        if self._view is not None:
            self._view.release()
            self._view = None
        if self._mapping is not None:
            self._mapping.flush()
            self._mapping.close()
            self._mapping = None

        self._file.truncate(new_capacity * self.record_bytes)
        self._file.flush()
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_WRITE)
        self._view = memoryview(self._mapping)
        self.capacity = new_capacity

    @staticmethod
    def _row_bytes(row: np.ndarray) -> memoryview:
        array = np.asarray(row, dtype=np.float64)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        return memoryview(array).cast("B")

    def append(self, row: np.ndarray) -> int:
        self._ensure_capacity()
        assert self._view is not None
        record_id = self.count
        start = record_id * self.record_bytes
        self._view[start : start + self.record_bytes] = self._row_bytes(row)
        self.count += 1
        return record_id

    def equals(self, record_id: int, row: np.ndarray) -> bool:
        assert self._view is not None and self._mapping is not None
        start = int(record_id) * self.record_bytes
        end = start + self.record_bytes
        candidate = self._row_bytes(row)
        stored = self._view[start:end]
        if stored == candidate:
            return True
        stored_values = np.frombuffer(
            self._mapping,
            dtype=np.float64,
            count=self.n_features,
            offset=start,
        )
        return bool(np.array_equal(stored_values, row, equal_nan=True))

    def close(self) -> None:
        if self._view is not None:
            self._view.release()
            self._view = None
        if self._mapping is not None:
            self._mapping.flush()
            self._mapping.close()
            self._mapping = None
        self._file.close()
        self.path.unlink(missing_ok=True)


def _find_feature_entry(
    bucket_entry: int,
    row: np.ndarray,
    collision_buckets: List[List[int]],
    payloads: _FeaturePayloadStore,
) -> Optional[int]:
    """Return the exact matching packed entry within one hash bucket."""
    if bucket_entry >= 0:
        record_id, _status = _unpack_feature_entry(bucket_entry)
        return bucket_entry if payloads.equals(record_id, row) else None

    bucket = collision_buckets[-bucket_entry - 1]
    for candidate in bucket:
        record_id, _status = _unpack_feature_entry(candidate)
        if payloads.equals(record_id, row):
            return candidate
    return None


def _resolve_feature_chunk(
    feature_hashes: np.ndarray,
    features: np.ndarray,
    sorted_hashes: np.ndarray,
    sorted_entries: np.ndarray,
    collision_buckets: List[List[int]],
    payloads: _FeaturePayloadStore,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve hash candidates to exact feature group ids and conflict flags."""
    positions = np.searchsorted(sorted_hashes, feature_hashes)
    if np.any(positions >= len(sorted_hashes)) or np.any(
        sorted_hashes[positions] != feature_hashes
    ):
        raise AccountingError("Pass replay produced a feature hash absent from pass 1")

    group_ids = np.empty(len(features), dtype=np.int64)
    conflicting = np.empty(len(features), dtype=bool)
    for i, position in enumerate(positions):
        bucket_entry = int(sorted_entries[int(position)])
        exact_entry = _find_feature_entry(
            bucket_entry, features[i], collision_buckets, payloads
        )
        if exact_entry is None:
            raise AccountingError(
                "Matching fingerprint bucket contained no exactly equal feature payload"
            )
        record_id, status = _unpack_feature_entry(exact_entry)
        group_ids[i] = record_id
        conflicting[i] = status == _FEATURE_CONFLICTING
    return group_ids, conflicting


# ---------------------------------------------------------------------------
# Collision-safe canonical cascade
# ---------------------------------------------------------------------------

def build_keep_index(
    csv_paths: List[Path],
    chunk_size: int = 200_000,
    row_limit: Optional[int] = None,
    progress: Optional[callable] = None,  # type: ignore[type-arg]
) -> CleaningResult:
    """Run the collision-safe canonical cascade and return the keep-index.

    Pass 1 builds exact Level-1 feature groups in fingerprint buckets.  Exact
    feature payloads live in a temporary disk-backed store.  Pass 2a identifies
    only the Level-2 and Level-3 hash buckets that repeat among non-conflicting
    rows.  Pass 2b applies the precedence cascade and compares exact canonical
    keys inside those repeated buckets.  No fingerprint is equality proof.
    """
    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    stats = CleaningStats(source_files=[p.name for p in csv_paths])
    payloads = _FeaturePayloadStore(len(MODEL_FEATURE_COLUMNS))
    try:
        # --------------------------------------------------------------
        # Pass 1 — exact Level-1 groups and per-group label status.
        # --------------------------------------------------------------
        _log("Cleaning pass 1/3: building exact Level-1 feature groups ...")
        feature_buckets: Dict[int, int] = {}
        collision_buckets: List[List[int]] = []

        for csv_path in csv_paths:
            _log(f"  pass 1 — {csv_path.name}")
            for chunk, counts in _iter_chunks(csv_path, chunk_size, row_limit):
                stats.raw_rows += counts.raw
                stats.embedded_header_rows += counts.embedded_header
                stats.unknown_label_rows += counts.unknown_label
                if chunk.empty:
                    continue

                features = chunk[list(MODEL_FEATURE_COLUMNS)].to_numpy(
                    dtype=np.float64
                )
                labels = np.array(
                    [LABEL_TO_INT[label] for label in chunk[LABEL_COLUMN]],
                    dtype=np.int16,
                )
                feature_hashes, _state_hashes, _exact_hashes = _cascade_hashes(
                    features,
                    chunk[TIMESTAMP_COLUMN].to_numpy(dtype=object),
                    labels,
                )

                for i, hash_value in enumerate(feature_hashes):
                    fingerprint = int(hash_value)
                    label = int(labels[i])
                    bucket_entry = feature_buckets.get(fingerprint)
                    if bucket_entry is None:
                        record_id = payloads.append(features[i])
                        feature_buckets[fingerprint] = _pack_feature_entry(
                            record_id, label
                        )
                        continue

                    if bucket_entry >= 0:
                        record_id, status = _unpack_feature_entry(bucket_entry)
                        if payloads.equals(record_id, features[i]):
                            if status != label and status != _FEATURE_CONFLICTING:
                                feature_buckets[fingerprint] = _pack_feature_entry(
                                    record_id, _FEATURE_CONFLICTING
                                )
                            continue

                        new_record = payloads.append(features[i])
                        collision_buckets.append(
                            [
                                bucket_entry,
                                _pack_feature_entry(new_record, label),
                            ]
                        )
                        feature_buckets[fingerprint] = -len(collision_buckets)
                        continue

                    collision_bucket = collision_buckets[-bucket_entry - 1]
                    matched_position: Optional[int] = None
                    for position, candidate in enumerate(collision_bucket):
                        record_id, _status = _unpack_feature_entry(candidate)
                        if payloads.equals(record_id, features[i]):
                            matched_position = position
                            break

                    if matched_position is None:
                        new_record = payloads.append(features[i])
                        collision_bucket.append(
                            _pack_feature_entry(new_record, label)
                        )
                    else:
                        candidate = collision_bucket[matched_position]
                        record_id, status = _unpack_feature_entry(candidate)
                        if status != label and status != _FEATURE_CONFLICTING:
                            collision_bucket[matched_position] = _pack_feature_entry(
                                record_id, _FEATURE_CONFLICTING
                            )

                stats.usable_rows += len(chunk)

        stats.conflicting_groups = 0
        for entry in feature_buckets.values():
            candidates = (
                [entry] if entry >= 0 else collision_buckets[-entry - 1]
            )
            stats.conflicting_groups += sum(
                _unpack_feature_entry(candidate)[1] == _FEATURE_CONFLICTING
                for candidate in candidates
            )

        n_distinct_features = payloads.count
        n_collision_hashes = len(collision_buckets)
        _log(
            f"  pass 1 done — {stats.raw_rows:,} raw rows, "
            f"{stats.usable_rows:,} usable, "
            f"{n_distinct_features:,} exact feature groups, "
            f"{stats.conflicting_groups:,} conflicting groups, "
            f"{n_collision_hashes:,} multi-payload fingerprint buckets"
        )

        # Compact the Python dictionary to sorted NumPy arrays before later
        # passes need the large Level-2/Level-3 seen sets.
        sorted_hashes = np.fromiter(
            feature_buckets.keys(), dtype=np.uint64, count=len(feature_buckets)
        )
        sorted_entries = np.fromiter(
            feature_buckets.values(), dtype=np.int64, count=len(feature_buckets)
        )
        order = np.argsort(sorted_hashes, kind="stable")
        sorted_hashes = sorted_hashes[order]
        sorted_entries = sorted_entries[order]
        del order, feature_buckets

        # --------------------------------------------------------------
        # Pass 2a — find only repeated L2/L3 fingerprint buckets.
        # --------------------------------------------------------------
        _log("Cleaning pass 2/3: locating repeated Level-2/Level-3 buckets ...")
        exact_once: Set[int] = set()
        exact_repeated: Set[int] = set()
        state_once: Set[int] = set()
        state_repeated: Set[int] = set()

        for csv_path in csv_paths:
            _log(f"  pass 2 — {csv_path.name}")
            for chunk, _counts in _iter_chunks(csv_path, chunk_size, row_limit):
                if chunk.empty:
                    continue
                features = chunk[list(MODEL_FEATURE_COLUMNS)].to_numpy(
                    dtype=np.float64
                )
                labels = np.array(
                    [LABEL_TO_INT[label] for label in chunk[LABEL_COLUMN]],
                    dtype=np.int16,
                )
                feature_hashes, state_hashes, exact_hashes = _cascade_hashes(
                    features,
                    chunk[TIMESTAMP_COLUMN].to_numpy(dtype=object),
                    labels,
                )
                _group_ids, conflicts = _resolve_feature_chunk(
                    feature_hashes,
                    features,
                    sorted_hashes,
                    sorted_entries,
                    collision_buckets,
                    payloads,
                )
                for i in range(len(features)):
                    if conflicts[i]:
                        continue
                    exact_hash = int(exact_hashes[i])
                    state_hash = int(state_hashes[i])
                    if exact_hash in exact_once:
                        exact_repeated.add(exact_hash)
                    else:
                        exact_once.add(exact_hash)
                    if state_hash in state_once:
                        state_repeated.add(state_hash)
                    else:
                        state_once.add(state_hash)

        del exact_once, state_once
        _log(
            f"  pass 2 done — {len(exact_repeated):,} repeated exact-hash "
            f"buckets, {len(state_repeated):,} repeated state-hash buckets"
        )

        # --------------------------------------------------------------
        # Pass 2b — exact cascade in deterministic source order.
        # --------------------------------------------------------------
        _log("Cleaning pass 3/3: applying exact mutually exclusive cascade ...")
        kept_per_file: List[Tuple[str, np.ndarray]] = []
        exact_payloads: Dict[int, List[Tuple[int, Optional[str], int]]] = {}
        state_payloads: Dict[int, List[Tuple[int, int]]] = {}

        for csv_path in csv_paths:
            _log(f"  pass 3 — {csv_path.name}")
            kept_indices: List[np.ndarray] = []
            for chunk, _counts in _iter_chunks(csv_path, chunk_size, row_limit):
                if chunk.empty:
                    continue
                features = chunk[list(MODEL_FEATURE_COLUMNS)].to_numpy(
                    dtype=np.float64
                )
                labels = np.array(
                    [LABEL_TO_INT[label] for label in chunk[LABEL_COLUMN]],
                    dtype=np.int16,
                )
                timestamps = chunk[TIMESTAMP_COLUMN].to_numpy(dtype=object)
                original_indices = chunk[ORIGINAL_INDEX_COLUMN].to_numpy(
                    dtype=np.int64
                )
                feature_hashes, state_hashes, exact_hashes = _cascade_hashes(
                    features, timestamps, labels
                )
                group_ids, conflicts = _resolve_feature_chunk(
                    feature_hashes,
                    features,
                    sorted_hashes,
                    sorted_entries,
                    collision_buckets,
                    payloads,
                )
                keep_mask = np.zeros(len(features), dtype=bool)

                for i in range(len(features)):
                    if conflicts[i]:
                        stats.conflicting_rows += 1
                        continue

                    group_id = int(group_ids[i])
                    label = int(labels[i])
                    exact_hash = int(exact_hashes[i])
                    exact_key = (
                        group_id,
                        _canonical_timestamp(timestamps[i]),
                        label,
                    )
                    if exact_hash in exact_repeated:
                        candidates = exact_payloads.setdefault(exact_hash, [])
                        if exact_key in candidates:
                            stats.exact_duplicate_rows_removed += 1
                            continue
                        candidates.append(exact_key)

                    keep_mask[i] = True
                    state_hash = int(state_hashes[i])
                    state_key = (group_id, label)
                    if state_hash in state_repeated:
                        candidates = state_payloads.setdefault(state_hash, [])
                        if state_key in candidates:
                            stats.temporal_duplicate_rows_retained += 1
                        else:
                            candidates.append(state_key)
                            stats.unique_rows += 1
                    else:
                        stats.unique_rows += 1

                kept_indices.append(original_indices[keep_mask])

            kept_per_file.append(
                (
                    csv_path.stem,
                    np.concatenate(kept_indices)
                    if kept_indices
                    else np.empty(0, dtype=np.int64),
                )
            )

        stats.clean_rows = sum(len(indices) for _, indices in kept_per_file)
        stats.validate()
        _log(
            f"  pass 3 done — conflicting {stats.conflicting_rows:,} | "
            f"exact removed {stats.exact_duplicate_rows_removed:,} | "
            f"temporal retained {stats.temporal_duplicate_rows_retained:,} | "
            f"unique {stats.unique_rows:,} | clean {stats.clean_rows:,}"
        )
        return CleaningResult(kept_per_file=kept_per_file, stats=stats)
    finally:
        payloads.close()


# ---------------------------------------------------------------------------
# Row-level classification (used by tests and small-scale diagnostics)
# ---------------------------------------------------------------------------

def classify_rows(
    features: np.ndarray,     # (N, 78) float64
    timestamps: List[str],
    labels: List[str],
) -> List[str]:
    """Classify an in-memory row set with the same cascade as the corpus pass.

    Returns one category string per row, in input order.  This is the
    executable specification of the cascade: it is deliberately simple, runs
    entirely in memory, and is what the mutual-exclusivity tests exercise.

    Input order is treated as the source order, so the first occurrence of a
    group is the one retained.
    """
    n = len(labels)
    if len(timestamps) != n or len(features) != n:
        raise ValueError("features, timestamps and labels must have equal length")

    codes = [LABEL_TO_INT[lbl] for lbl in labels]
    feat_hashes_arr, featlabel_hashes, exact_hashes = _cascade_hashes(
        features,
        np.asarray(timestamps, dtype=object),
        np.asarray(codes, dtype=np.int16),
    )
    feat_hashes = [int(value) for value in feat_hashes_arr]

    # Level 1: a fingerprint selects a candidate bucket.  Exact feature values
    # select the group within that bucket.
    groups: List[Tuple[np.ndarray, Set[int]]] = []
    groups_by_hash: Dict[int, List[int]] = {}
    group_for_row: List[int] = []
    for i, (feature_hash, label) in enumerate(zip(feat_hashes, codes)):
        group_id: Optional[int] = None
        for candidate_id in groups_by_hash.get(feature_hash, []):
            candidate, _candidate_labels = groups[candidate_id]
            if np.array_equal(candidate, features[i], equal_nan=True):
                group_id = candidate_id
                break
        if group_id is None:
            group_id = len(groups)
            groups.append((np.array(features[i], copy=True), {label}))
            groups_by_hash.setdefault(feature_hash, []).append(group_id)
        else:
            groups[group_id][1].add(label)
        group_for_row.append(group_id)

    conflicting_groups = {
        group_id for group_id, (_payload, group_labels) in enumerate(groups)
        if len(group_labels) > 1
    }

    out: List[str] = []
    exact_seen: Dict[int, List[Tuple[int, Optional[str], int]]] = {}
    featlabel_seen: Dict[int, List[Tuple[int, int]]] = {}

    for i in range(n):
        group_id = group_for_row[i]
        h_fl = int(featlabel_hashes[i])
        h_exact = int(exact_hashes[i])
        label = codes[i]
        exact_key = (group_id, _canonical_timestamp(timestamps[i]), label)
        state_key = (group_id, label)

        if group_id in conflicting_groups:
            out.append(CATEGORY_CONFLICTING)
        elif exact_key in exact_seen.setdefault(h_exact, []):
            out.append(CATEGORY_EXACT_DUPLICATE)
        else:
            exact_seen[h_exact].append(exact_key)
            if state_key in featlabel_seen.setdefault(h_fl, []):
                out.append(CATEGORY_TEMPORAL_DUPLICATE)
            else:
                featlabel_seen[h_fl].append(state_key)
                out.append(CATEGORY_UNIQUE)

    return out
