"""T=5 non-overlapping temporal window construction.

Each window contains exactly T consecutive flow records from the globally-
ordered, cleaned capture-order sequence.  The label of a window is the label
of its *last* (T-th) flow — not the majority label.  This matches the
HTGraph convention where the final node in the window represents the current
state being classified.

Ordering
--------
Flows are ordered by the deterministic capture-order proxy:

    (source_file_capture_date, Original_Index)

Timestamp is loaded and stored as a diagnostic column but is NOT used as the
primary sort key.  The 12-hour clock without AM/PM markers makes Timestamp
ambiguous; treating capture order as the proxy avoids reconstructing AM/PM.

Window boundaries are strict: a window of T flows is discarded rather than
padded if fewer than T flows remain at the end of the sequence.  Windows never
span file boundaries — the sequence is already global, so a window *can* span
files naturally; the only discard is at the very end of the global sequence
when ``total_clean_rows % T != 0``.

Outputs
-------
:func:`build_windows` returns:

* ``X``   — float32 array of shape ``(n_windows, T, 78)``
* ``y``   — int16  array of shape ``(n_windows,)``  (label of the last flow)
* ``meta`` — list of dicts with per-window metadata for diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.labels import LABEL_TO_INT, encode_labels
from src.preprocessing.schema import (
    EMBEDDED_HEADER_LABEL,
    LABEL_COLUMN,
    MODEL_FEATURE_COLUMNS,
    ORIGINAL_INDEX_COLUMN,
    TIMESTAMP_COLUMN,
)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class WindowingResult:
    """Output of :func:`build_windows`."""

    #: Feature tensor: shape (n_windows, T, 78), float32.
    X: np.ndarray
    #: Label vector: shape (n_windows,), int16.  Label = label(flow T).
    y: np.ndarray
    #: Per-window metadata: one dict per window.  Built lazily on access
    #: (see :class:`_WindowMetaView`) — supports ``len()``, indexing and
    #: iteration without materialising a dict per window.
    meta: Sequence[Dict[str, object]]
    #: Summary statistics.
    stats: "WindowingStats"


@dataclass
class WindowingStats:
    """Aggregate counts from the windowing pass."""

    window_size: int = 5
    total_clean_rows: int = 0
    n_windows: int = 0
    #: Rows at the tail of the sequence that were discarded because
    #: ``total_clean_rows % T != 0``.
    tail_rows_discarded: int = 0
    #: Class distribution: label_string → window count.
    class_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "window_size": self.window_size,
            "total_clean_rows": self.total_clean_rows,
            "n_windows": self.n_windows,
            "tail_rows_discarded": self.tail_rows_discarded,
            "label_source": "label of the last (T-th) flow in each window",
            "class_distribution": self.class_distribution,
        }


# ---------------------------------------------------------------------------
# Stream assembly
# ---------------------------------------------------------------------------

def _is_sorted(a: np.ndarray) -> bool:
    """True if *a* is non-decreasing."""
    return True if len(a) < 2 else bool(np.all(a[:-1] <= a[1:]))


def _fill_kept_rows(
    csv_path: Path,
    kept_indices: np.ndarray,
    chunk_size: int,
    feature_out: np.ndarray,
    label_out: np.ndarray,
    offset: int,
) -> int:
    """Stream *csv_path* and write its kept rows into pre-allocated arrays.

    Rows land at ``feature_out[offset : offset + k]`` (float32) and
    ``label_out[offset : offset + k]`` (int16) in ascending Original_Index
    order, which is the file's capture order.  Only one chunk is materialised
    at a time, so the cleaned corpus never exists as a single DataFrame — that
    form cost ~8.9 GB as float64 plus a second copy at concat time.

    Embedded header rows need no special handling here: the cleaning pass
    stripped them before assigning kept indices, so they can never be selected.

    Returns the number of rows written (k).
    """
    if len(kept_indices) == 0:
        return 0

    # searchsorted membership below requires ascending order.  The cleaning
    # pass emits ascending indices; enforce rather than assume.
    kept_sorted = kept_indices if _is_sorted(kept_indices) else np.sort(kept_indices)
    last_kept_index = int(kept_sorted[-1])

    usecols = list(MODEL_FEATURE_COLUMNS) + [LABEL_COLUMN]
    feat_cols = list(MODEL_FEATURE_COLUMNS)

    written = 0
    row_offset = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=str,          # embedded header rows would break a float cast
        chunksize=chunk_size,
        low_memory=False,
    ):
        n = len(chunk)
        idx = np.arange(row_offset, row_offset + n, dtype=np.int64)
        row_offset += n

        # Vectorised membership test against the kept set.
        pos = np.searchsorted(kept_sorted, idx)
        np.clip(pos, 0, len(kept_sorted) - 1, out=pos)
        mask = kept_sorted[pos] == idx
        if not mask.any():
            continue

        sub = chunk.iloc[np.flatnonzero(mask)]
        k = len(sub)
        block = feature_out[offset + written : offset + written + k]
        for j, col in enumerate(feat_cols):
            block[:, j] = pd.to_numeric(sub[col], errors="coerce").to_numpy(
                dtype=np.float32
            )
        label_out[offset + written : offset + written + k] = (
            sub[LABEL_COLUMN].map(LABEL_TO_INT).to_numpy(dtype=np.int16)
        )
        written += k

        # In row-limited/debug runs every requested index may be near the head
        # of a multi-gigabyte CSV.  Once the current raw chunk has passed the
        # largest kept Original_Index, no later chunk can contain a match.
        if row_offset > last_kept_index:
            break

    return written


class _WindowMetaView:
    """Lazy read-only sequence of per-window metadata dicts.

    Materialising one dict per window cost ~1 GB at corpus scale while holding
    nothing that is not already in ``y``.  This view builds each dict on access
    instead, preserving the ``len()`` / ``[i]`` interface callers use.
    """

    __slots__ = ("_y",)

    def __init__(self, y: np.ndarray) -> None:
        self._y = y

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, i: int) -> Dict[str, object]:
        from src.preprocessing.labels import INT_TO_LABEL

        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self._y)))]
        if i < 0:
            i += len(self._y)
        if not 0 <= i < len(self._y):
            raise IndexError(i)
        code = int(self._y[i])
        return {
            "window_idx": i,
            "label_code": code,
            "label_str": INT_TO_LABEL[code],
        }

    def __iter__(self):
        for i in range(len(self._y)):
            yield self[i]


# ---------------------------------------------------------------------------
# Main windowing function
# ---------------------------------------------------------------------------

def build_windows(
    csv_paths: List[Path],
    kept_per_file: List[Tuple[str, np.ndarray]],
    window_size: int = 5,
    chunk_size: int = 200_000,
    progress: Optional[callable] = None,  # type: ignore[type-arg]
    work_dir: Optional[Path] = None,
) -> WindowingResult:
    """Construct non-overlapping T=*window_size* windows from the cleaned corpus.

    Parameters
    ----------
    csv_paths:
        Day CSV files in deterministic capture-date order.  Must match the
        order used in :func:`src.preprocessing.cleaning.build_keep_index`.
    kept_per_file:
        List of ``(file_stem, original_index_array)`` produced by the cleaning
        pass.  The order must match *csv_paths*.
    window_size:
        Number of consecutive flows per window (T).
    chunk_size:
        Rows per ``read_csv`` chunk when loading kept rows.
    progress:
        Optional callable for status messages.
    work_dir:
        When supplied, allocate the assembled feature/label arrays as
        disk-backed ``.npy`` memmaps in this directory.  This is the full-data
        path and prevents the 4.53-GiB unscaled tensor from exhausting RAM.
    """
    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    T = window_size

    # Map file stem → kept index array for lookup.
    stem_to_kept: Dict[str, np.ndarray] = {
        stem: arr for stem, arr in kept_per_file
    }

    # ------------------------------------------------------------------
    # Assemble the global feature matrix and label vector in capture order.
    #
    # Pre-allocate and fill in place.  The previous form built one DataFrame
    # per file, concatenated them, then cast to float32 — that held the corpus
    # as float64 (~8.9 GB) plus a concat destination plus the float32 copy, and
    # exhausted memory before it could return.
    # ------------------------------------------------------------------
    n_features = len(MODEL_FEATURE_COLUMNS)
    total_kept = sum(len(arr) for _, arr in kept_per_file)

    _log("Windowing: assembling global sequence from cleaned rows ...")
    _log(
        f"  allocating {total_kept:,} x {n_features} float32 "
        f"({total_kept * n_features * 4 / 1024 ** 3:.2f} GB)"
    )

    if work_dir is None:
        feature_arr = np.empty((total_kept, n_features), dtype=np.float32)
        label_codes = np.empty(total_kept, dtype=np.int16)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        feature_arr = np.lib.format.open_memmap(
            work_dir / "unscaled_flow_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_kept, n_features),
        )
        label_codes = np.lib.format.open_memmap(
            work_dir / "flow_labels.npy",
            mode="w+",
            dtype=np.int16,
            shape=(total_kept,),
        )

    offset = 0
    for csv_path in csv_paths:
        stem = csv_path.stem
        kept = stem_to_kept.get(stem, np.empty(0, dtype=np.int64))
        _log(f"  loading {csv_path.name} — {len(kept):,} kept rows")
        k = _fill_kept_rows(
            csv_path, kept, chunk_size, feature_arr, label_codes, offset
        )
        # A mismatch means the Original_Index proxy disagreed between the
        # cleaning pass and this pass — the window ordering would be wrong.
        if k != len(kept):
            raise RuntimeError(
                f"{csv_path.name}: cleaning kept {len(kept):,} rows but "
                f"windowing matched {k:,}; Original_Index proxy is inconsistent"
            )
        offset += k

    total_clean_rows = offset
    if isinstance(feature_arr, np.memmap):
        feature_arr.flush()
    if isinstance(label_codes, np.memmap):
        label_codes.flush()
    _log(f"  global sequence: {total_clean_rows:,} rows")

    # ------------------------------------------------------------------
    # Slice into non-overlapping windows.
    # ------------------------------------------------------------------
    n_windows = total_clean_rows // T
    tail = total_clean_rows - n_windows * T

    _log(
        f"  {total_clean_rows:,} rows ÷ T={T} = {n_windows:,} windows "
        f"({tail} tail rows discarded)"
    )

    # shape: (n_windows, T, 78)
    X = feature_arr[: n_windows * T].reshape(n_windows, T, len(MODEL_FEATURE_COLUMNS))
    # Label = last flow in each window.
    y = label_codes[T - 1 : n_windows * T : T]   # every T-th starting at T-1

    # ------------------------------------------------------------------
    # Per-window metadata.  Lazy — one dict per window would cost ~1 GB at
    # corpus scale while holding nothing that is not already in y.
    # ------------------------------------------------------------------
    meta = _WindowMetaView(y)

    # ------------------------------------------------------------------
    # Class distribution.
    # ------------------------------------------------------------------
    from src.preprocessing.labels import INT_TO_LABEL as _ITL
    class_dist: Dict[str, int] = {}
    unique_codes, counts = np.unique(y, return_counts=True)
    for code, cnt in zip(unique_codes, counts):
        class_dist[_ITL[int(code)]] = int(cnt)

    stats = WindowingStats(
        window_size=T,
        total_clean_rows=total_clean_rows,
        n_windows=n_windows,
        tail_rows_discarded=tail,
        class_distribution=class_dist,
    )

    return WindowingResult(X=X, y=y, meta=meta, stats=stats)
