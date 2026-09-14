"""Isolated, read-only diagnostic analyses.

Nothing in this package is part of the training pipeline. Each module answers
one empirical question about the raw corpus and writes its own artifacts under
``results/``. They import from :mod:`src.preprocessing` but never modify it, and
they never write under ``data/raw``.

* :mod:`src.analysis.window_continuity` - measures whether T=5 windows of
  globally consecutive capture-order flows share a communication context.
"""
