"""Release-only authority regressions using compact temporary fixtures."""
from pathlib import Path
import hashlib
import json

import pytest

import authority_fixture_support as support
from src.pruning import stage_e12b as b

ROOT = Path(__file__).resolve().parents[1]


def closure_fixture(root):
    paths = (support.CLOSURE_BINDING_PATH, "PROJECT_HANDOFF.md", b.AUTHORITY_PATH,
             str(Path(b.AUTHORITY_PATH).with_suffix(".sha256")),
             "tests/fixtures/reconciled_execution_handoff.md")
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())


def test_closure_and_execution_handoffs_have_separate_authenticated_bindings(tmp_path):
    closure_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    snapshot = support.reconciled_execution_handoff(tmp_path)
    binding = json.loads((tmp_path / support.CLOSURE_BINDING_PATH).read_bytes())
    assert hashlib.sha256(snapshot).hexdigest() == b.CURRENT_HANDOFF_SHA256
    assert binding["historical_e9_handoff_sha256"] == b.HISTORICAL_HANDOFF_SHA256
    assert binding["closure_handoff_sha256"] != b.CURRENT_HANDOFF_SHA256
    assert json.loads((tmp_path / b.AUTHORITY_PATH).read_bytes())["current_handoff_sha256"] == b.CURRENT_HANDOFF_SHA256
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("relative, message", (
    ("PROJECT_HANDOFF.md", "closure handoff"),
    ("tests/fixtures/reconciled_execution_handoff.md", "execution handoff"),
    (b.AUTHORITY_PATH, "reconciliation attestation"),
    (str(Path(b.AUTHORITY_PATH).with_suffix(".sha256")), "sidecar"),
    (support.CLOSURE_BINDING_PATH, "closure binding"),
))
def test_authority_epoch_corruption_is_rejected(tmp_path, relative, message):
    closure_fixture(tmp_path)
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match=message):
        support.reconciled_execution_handoff(tmp_path)


def test_closure_handoff_does_not_relax_frozen_execution_gate(tmp_path, monkeypatch):
    import test_stage_e12b as frozen
    # This module receives no fixture override: the original frozen setup copies
    # today's closure file, which must remain rejected by the execution gate.
    frozen._authority_fixture(tmp_path, monkeypatch)
    with b.file_guard(tmp_path) as access, pytest.raises(b.ComparisonError, match="live authority handoff changed"):
        b.verify_authority_reconciliation(tmp_path)
    assert not access.test_enabled and not access.opened_test
