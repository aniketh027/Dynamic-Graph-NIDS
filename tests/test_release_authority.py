"""Release-only authority regressions using compact temporary fixtures."""
from pathlib import Path
import hashlib
import json

import pytest

import authority_fixture_support as support
from src.pruning import stage_e12b as b

ROOT = Path(__file__).resolve().parents[1]
CLOSURE_HANDOFF_PATH = "tests/fixtures/scientific_closure_handoff.md"
CLOSURE_HANDOFF_SHA256 = "cdbdf14742b9f7a7a7b982305d97e2b39d53aa652672c8a6f03ba657d44df230"
EXECUTION_HANDOFF_PATH = "tests/fixtures/reconciled_execution_handoff.md"


def closure_fixture(root):
    paths = (support.CLOSURE_BINDING_PATH, CLOSURE_HANDOFF_PATH, b.AUTHORITY_PATH,
             str(Path(b.AUTHORITY_PATH).with_suffix(".sha256")),
             EXECUTION_HANDOFF_PATH)
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())


def test_historical_closure_snapshot_matches_independent_release_identity(tmp_path):
    closure_fixture(tmp_path)
    snapshot = (tmp_path / CLOSURE_HANDOFF_PATH).read_bytes()
    assert len(snapshot) == 60183
    assert hashlib.sha256(snapshot).hexdigest() == CLOSURE_HANDOFF_SHA256
    assert hashlib.sha1(b"blob " + str(len(snapshot)).encode("ascii") + b"\0" + snapshot).hexdigest() == "c865a67236b45dd150e4f4d0973507278debe5cd"
    assert snapshot == b.subprocess.check_output(
        ["git", "cat-file", "blob", "2854d153d65c93689875884f3ad45f0b3b468ec2:PROJECT_HANDOFF.md"],
        cwd=ROOT, timeout=10)


def test_closure_and_execution_handoffs_have_separate_authenticated_bindings(tmp_path):
    closure_fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    snapshot = support.reconciled_execution_handoff(tmp_path)
    binding = json.loads((tmp_path / support.CLOSURE_BINDING_PATH).read_bytes())
    assert snapshot == (ROOT / EXECUTION_HANDOFF_PATH).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == b.CURRENT_HANDOFF_SHA256
    assert binding["historical_e9_handoff_sha256"] == b.HISTORICAL_HANDOFF_SHA256
    assert binding["closure_handoff_sha256"] == CLOSURE_HANDOFF_SHA256
    assert binding["closure_handoff_sha256"] != b.CURRENT_HANDOFF_SHA256
    assert json.loads((tmp_path / b.AUTHORITY_PATH).read_bytes())["current_handoff_sha256"] == b.CURRENT_HANDOFF_SHA256
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("live_state", ("absent", "replacement", "addendum", "application"))
def test_release_helper_is_independent_of_live_application_handoff(tmp_path, live_state):
    closure_fixture(tmp_path)
    live = tmp_path / "PROJECT_HANDOFF.md"
    if live_state == "replacement":
        live.write_bytes(b"Ordinary current application status.\n")
    elif live_state == "addendum":
        live.write_bytes((ROOT / "PROJECT_HANDOFF.md").read_bytes() + b"\nCurrent application addendum.\n")
    elif live_state == "application":
        live.write_bytes((ROOT / "PROJECT_HANDOFF.md").read_bytes())
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert support.reconciled_execution_handoff(tmp_path) == (ROOT / EXECUTION_HANDOFF_PATH).read_bytes()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("relative, message", (
    (CLOSURE_HANDOFF_PATH, "closure handoff"),
    (EXECUTION_HANDOFF_PATH, "execution handoff"),
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


def test_missing_historical_closure_snapshot_is_rejected(tmp_path):
    closure_fixture(tmp_path)
    (tmp_path / CLOSURE_HANDOFF_PATH).unlink()
    with pytest.raises(FileNotFoundError, match="scientific_closure_handoff"):
        support.reconciled_execution_handoff(tmp_path)


@pytest.mark.parametrize("change", ("bom", "extra_lf", "missing_final_lf", "crlf"))
def test_closure_snapshot_encoding_and_newline_changes_are_rejected(tmp_path, change):
    closure_fixture(tmp_path)
    target = tmp_path / CLOSURE_HANDOFF_PATH
    original = target.read_bytes()
    changed = {
        "bom": b"\xef\xbb\xbf" + original,
        "extra_lf": original + b"\n",
        "missing_final_lf": original.removesuffix(b"\n"),
        "crlf": original.replace(b"\n", b"\r\n"),
    }[change]
    assert changed != original
    target.write_bytes(changed)
    with pytest.raises(ValueError, match="closure handoff changed"):
        support.reconciled_execution_handoff(tmp_path)


@pytest.mark.parametrize("epoch", ("historical_e9", "execution", "application"))
def test_other_authority_epochs_cannot_substitute_for_scientific_closure(tmp_path, epoch):
    closure_fixture(tmp_path)
    if epoch == "historical_e9":
        substitute = b.subprocess.check_output(
            ["git", "cat-file", "blob", f"{b.E9_FROZEN_COMMIT}:PROJECT_HANDOFF.md"],
            cwd=ROOT, timeout=10)
        assert hashlib.sha256(substitute).hexdigest() == b.HISTORICAL_HANDOFF_SHA256
    else:
        relative = EXECUTION_HANDOFF_PATH if epoch == "execution" else "PROJECT_HANDOFF.md"
        substitute = (ROOT / relative).read_bytes()
    assert hashlib.sha256(substitute).hexdigest() != CLOSURE_HANDOFF_SHA256
    (tmp_path / CLOSURE_HANDOFF_PATH).write_bytes(substitute)
    with pytest.raises(ValueError, match="closure handoff changed"):
        support.reconciled_execution_handoff(tmp_path)


@pytest.mark.parametrize("field", (
    "historical_e9_handoff_sha256", "reconciled_execution_handoff_sha256",
    "reconciliation_attestation_sha256", "closure_handoff_sha256",
    "execution_handoff_fixture",
))
def test_coordinated_changes_cannot_bypass_independent_epoch_pins(tmp_path, monkeypatch, field):
    closure_fixture(tmp_path)
    path = tmp_path / support.CLOSURE_BINDING_PATH
    binding = json.loads(path.read_bytes())
    targets = {
        "closure_handoff_sha256": CLOSURE_HANDOFF_PATH,
        "reconciled_execution_handoff_sha256": EXECUTION_HANDOFF_PATH,
        "reconciliation_attestation_sha256": b.AUTHORITY_PATH,
    }
    if field in targets:
        target = tmp_path / targets[field]
        target.write_bytes(target.read_bytes() + b"coordinated corruption")
        binding[field] = hashlib.sha256(target.read_bytes()).hexdigest()
        if field == "reconciliation_attestation_sha256":
            target.with_suffix(".sha256").write_text(binding[field], encoding="ascii")
    elif field == "execution_handoff_fixture":
        substitute = tmp_path / "substituted_execution.md"
        substitute.write_bytes((tmp_path / EXECUTION_HANDOFF_PATH).read_bytes())
        binding[field] = substitute.name
    else:
        binding[field] = "0" * 64
    path.write_bytes(json.dumps(binding, sort_keys=True).encode("utf-8"))
    with pytest.raises(ValueError, match="closure binding changed"):
        support.reconciled_execution_handoff(tmp_path)
    # Even rebinding the outer physical digest cannot replace the independent
    # epoch/content pins. All production validator constants remain untouched.
    monkeypatch.setattr(support, "CLOSURE_BINDING_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="authority epoch binding differs"):
        support.reconciled_execution_handoff(tmp_path)


@pytest.mark.parametrize("epoch", ("scientific_closure", "application", "future"))
def test_closure_handoff_does_not_relax_frozen_execution_gate(tmp_path, monkeypatch, epoch):
    import test_stage_e12b as frozen
    # This module receives no conftest override. Only temporary handoff bytes
    # change; the original frozen execution validator and its pins stay intact.
    frozen._authority_fixture(tmp_path, monkeypatch)
    if epoch == "scientific_closure":
        (tmp_path / "PROJECT_HANDOFF.md").write_bytes((ROOT / CLOSURE_HANDOFF_PATH).read_bytes())
    elif epoch == "future":
        live = tmp_path / "PROJECT_HANDOFF.md"
        live.write_bytes(live.read_bytes() + b"\nFuture application addendum.\n")
    with b.file_guard(tmp_path) as access, pytest.raises(b.ComparisonError, match="live authority handoff changed"):
        b.verify_authority_reconciliation(tmp_path)
    assert not access.test_enabled and not access.opened_test


def test_authenticated_execution_fixture_still_passes_frozen_execution_gate(tmp_path, monkeypatch):
    import test_stage_e12b as frozen
    attestation, _ = frozen._authority_fixture(tmp_path, monkeypatch)
    (tmp_path / "PROJECT_HANDOFF.md").write_bytes(support.reconciled_execution_handoff(ROOT))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with b.file_guard(tmp_path) as access:
        assert b.verify_authority_reconciliation(tmp_path) == attestation
        assert not access.test_enabled and not access.opened_test and not access.attempts
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
