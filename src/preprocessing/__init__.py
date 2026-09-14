"""Dataset inspection, cleaning and splitting.

Implemented so far:

* :mod:`src.preprocessing.inspection` - schema-agnostic dataset inspection.
* :mod:`src.preprocessing.schema` - the confirmed CSE-CIC-IDS2018 CSV schema
  (80 columns; 76 numeric flow features + Dst Port + Protocol = 78 model
  features; Timestamp and Label are not model features).
* :mod:`src.preprocessing.loader` - raw day-CSV loader producing a validated,
  fully finite feature frame plus string labels, parsed timestamps and the
  original row positions.
* :mod:`src.preprocessing.profiling` - Stage B multi-day corpus profiler:
  discovers every raw CSV, sequences the files by capture date and measures row
  accounting, labels, non-finite values, negatives, duplicates and timestamps
  in bounded memory. Measure-only - it cleans, replaces and removes nothing.
* :mod:`src.preprocessing.profile_artifacts` - JSON/CSV/Markdown views of a
  profiling run, written under the configured results directory.

Stage C — preprocessing, windowing, splitting and scaling (implemented):

* :mod:`src.preprocessing.labels` - fixed 15-label vocabulary, encode/decode
  helpers, class-mapping dict with absent-class annotations.
* :mod:`src.preprocessing.cleaning` - two-pass FNV-1a global duplicate cleaning:
  excludes Category A (exact) and Category C (conflicting) rows; retains
  Category B (temporal duplicates).
* :mod:`src.preprocessing.windowing` - T=5 non-overlapping window construction
  in deterministic capture-order; label = label of the last flow in each window.
* :mod:`src.preprocessing.transforms` - RobustScaler fit on E1 train only,
  applied to all six splits; -1 sentinel preservation in Init Fwd/Bwd Win Byts.
* :mod:`src.preprocessing.splitting` - Experiment 1 (global chronological
  70/10/20) and Experiment 2 (per-class chronological 70/10/20) splits with
  rare-class documentation.
* :mod:`src.preprocessing.artifacts` - writes .npy arrays, JSON metadata and
  summary artefacts under data/processed/preprocessed/.
* :mod:`src.preprocessing.pipeline` - top-level orchestrator (8 steps).

Graph construction, HTGraph, GATv2, HTGNN, pruning, training and inference
are later stages and are not implemented here.
"""
