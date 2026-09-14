"""Reproducibility helpers.

Only the seeding that the current phase actually needs (``random`` + ``numpy``).
Torch seeding is added in the modelling phase, together with the torch
dependency, so this module stays importable without a deep-learning stack.
"""

from __future__ import annotations

import os
import random
from typing import List

import numpy as np


def set_seed(seed: int) -> List[str]:
    """Seed Python's ``random``, ``numpy`` and ``PYTHONHASHSEED``.

    Returns the names of the components that were actually seeded, so callers
    can log exactly what reproducibility guarantees are in effect.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return ["python.random", "numpy.random", "PYTHONHASHSEED"]
