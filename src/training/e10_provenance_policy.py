"""Reviewed E.10 aggregation policy; never inferred from the live source hash.

This data-only module is part of the trusted result validator, not the training
implementation. Keep the approved runtime digests here to avoid making
full_training.py contain its own digest. Adding a revision requires source
review and regression verification; a version label alone is not approval.
"""

from typing import Final


AUDITED_STUDY: Final = "e10-official-20261001-v1"
AUDITED_WINDOWS_PLATFORMS: Final = (
    "Windows-11-10.0.26200-SP0",
    "Windows-11-10.0.26300-SP0",
)
TRAINING_SEMANTICS_ID: Final = "e10-v2-reviewed-training-semantics-20261003"
HISTORICAL_RUNTIME_SHA256: Final = (
    "b1aac183565c08e0d8d5307d466f5e3a60762f6604e9390bfddd3a54120f4023",
    "52222fb503466aee88e79f88c7defd7e12d90891d75ec507317a1bc8f25bb7db",
)
# Exact post-correction runtime bytes; changes require a new explicit review.
CORRECTED_RUNTIME_SHA256: Final = (
    "5bac1f7b37b718230604aebfdf035a0f42c25e99017d3dbd0e737eaa0d89fd77"
)
APPROVED_RUNTIME_SHA256: Final = (
    *HISTORICAL_RUNTIME_SHA256,
    CORRECTED_RUNTIME_SHA256,
)
APPROVED_RUNNER_SHA256: Final = (
    "3313c700240f88358d5a0de742ac3d51ef53bc9e043eecb91d003aacdbfa02c5"
)
