"""Post-completion fixture bindings; no change to frozen execution validators."""
from pathlib import Path
import hashlib
import json

from src.pruning import stage_e12b as b

CLOSURE_BINDING_PATH = "release/handoff_closure_binding.json"
CLOSURE_BINDING_SHA256 = "ddfe421dacbc9397ae6c7b2cdccdedbf2f0d2a8cd925343eab13d7a2ec36fa8f"


def reconciled_execution_handoff(root: Path) -> bytes:
    """Validate the three authority epochs and return the execution-era bytes."""
    root = Path(root)
    binding_bytes = (root / CLOSURE_BINDING_PATH).read_bytes()
    if hashlib.sha256(binding_bytes).hexdigest() != CLOSURE_BINDING_SHA256:
        raise ValueError("closure binding changed")
    binding = json.loads(binding_bytes)
    expected = {
        "historical_e9_handoff_sha256": b.HISTORICAL_HANDOFF_SHA256,
        "reconciled_execution_handoff_sha256": b.CURRENT_HANDOFF_SHA256,
        "reconciliation_attestation_sha256": b.AUTHORITY_SHA256,
        "closure_handoff_sha256": "cdbdf14742b9f7a7a7b982305d97e2b39d53aa652672c8a6f03ba657d44df230",
        "execution_handoff_fixture": "tests/fixtures/reconciled_execution_handoff.md",
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("authority epoch binding differs")
    for relative, expected_hash, label in (
        ("tests/fixtures/scientific_closure_handoff.md", binding["closure_handoff_sha256"], "closure handoff"),
        (b.AUTHORITY_PATH, b.AUTHORITY_SHA256, "reconciliation attestation"),
        (binding["execution_handoff_fixture"], b.CURRENT_HANDOFF_SHA256, "execution handoff"),
    ):
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{label} changed")
    if (root / b.AUTHORITY_PATH).with_suffix(".sha256").read_text(encoding="ascii").strip() != b.AUTHORITY_SHA256:
        raise ValueError("reconciliation attestation sidecar changed")
    return (root / binding["execution_handoff_fixture"]).read_bytes()
