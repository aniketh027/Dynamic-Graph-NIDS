"""Synthetic C3A tests; all files belong to an external temporary directory."""
import csv
import io
import json
import os
from pathlib import Path
import _winapi

import numpy as np
import pytest

from backend.app.services import csv_compatibility as compat, csv_replay
from backend.prepare_flow_csv import main
from src.graph.schema import canonical_sha256
from src.preprocessing.schema import CSV_COLUMNS, CSV_COLUMNS_EXTENDED, MODEL_FEATURE_COLUMNS


def encoded(header, rows=(), *, newline="\n", quoted=False, bom=False):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator=newline, quoting=csv.QUOTE_ALL if quoted else csv.QUOTE_MINIMAL)
    writer.writerow(header)
    writer.writerows(rows)
    return (b"\xef\xbb\xbf" if bom else b"") + text.getvalue().encode()


def source_rows(header, n=13):
    rows = []
    for i in range(n):
        rows.append([str(i * 1000 + MODEL_FEATURE_COLUMNS.index(name))
                     if name in MODEL_FEATURE_COLUMNS else f"private-{name}-{n-i},\"secret\"\nline"
                     for name in header])
    return rows


def files(tmp_path, *, header=CSV_COLUMNS, rows=None, payload=None):
    source, output = tmp_path / "input.csv", tmp_path / "prepared.csv"
    source.write_bytes(payload if payload is not None else encoded(header, source_rows(header) if rows is None else rows))
    return source, output


def decoded(payload):
    return list(csv.reader(io.StringIO(payload.decode(), newline=""), strict=True))


def assert_no_results(tmp_path, source):
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("header", [CSV_COLUMNS, CSV_COLUMNS_EXTENDED])
@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("quoted", [False, True])
def test_projection_preservation_and_parser_equivalence(tmp_path, header, bom, newline, quoted):
    rows = source_rows(header)
    rows[3] = rows[2].copy()  # duplicate-looking records remain
    source, output = files(tmp_path, payload=encoded(header, rows, bom=bom, newline=newline, quoted=quoted))
    original = source.read_bytes()
    expected = [[row[header.index(name)] for name in MODEL_FEATURE_COLUMNS] for row in rows]
    before = set(tmp_path.iterdir())
    preview = compat.preview(source)
    assert set(tmp_path.iterdir()) == before and source.read_bytes() == original
    assert preview["source_column_count"] == len(header) and preview["output_feature_count"] == 78
    assert preview["source_profile"] == ("standard_80" if len(header) == 80 else "extended_84")
    assert preview["further_record_validation_required"] and not preview["exporter_authenticated"]
    manifest = compat.prepare(source, output)
    payload = output.read_bytes()
    assert decoded(payload) == [list(MODEL_FEATURE_COLUMNS), *expected]
    parsed = csv_replay.parse_csv(io.BytesIO(payload))
    direct = csv_replay.parse_csv(io.BytesIO(encoded(MODEL_FEATURE_COLUMNS, expected)))
    np.testing.assert_array_equal(parsed.rows, direct.rows)
    assert not parsed.columns_reordered and len(parsed.rows) == 13
    assert parsed.window_count == 2 and len(parsed.rows) % 5 == 3
    for k, target in enumerate(range(4, len(rows) - len(rows) % 5, 5)):
        np.testing.assert_array_equal(parsed.rows.reshape(-1, 78)[k*5+4], direct.rows[target])
        assert expected[target][0] == str(target * 1000)
    assert source.read_bytes() == original
    assert manifest["source_record_count"] == manifest["output_record_count"] == 13
    assert manifest["rows_removed"] == 0
    assert compat.verify(output) == manifest
    assert manifest["source_to_model_columns"] == [
        {"source_index": header.index(name), "model_index": i, "name": name}
        for i, name in enumerate(MODEL_FEATURE_COLUMNS)]
    assert manifest["feature_order_sha256"] == canonical_sha256(MODEL_FEATURE_COLUMNS)
    assert manifest["removed_metadata_fields"] == [n for n in header if n not in MODEL_FEATURE_COLUMNS]
    assert all(manifest[k] is False for k in ("units_changed", "values_imputed", "transforms_applied"))
    assert "private-" not in payload.decode() + compat.sidecar_path(output).read_text()
    assert str(tmp_path) not in compat.sidecar_path(output).read_text()


@pytest.mark.parametrize("header", [CSV_COLUMNS, CSV_COLUMNS_EXTENDED])
def test_numeric_text_and_sentinels_are_not_reformatted(tmp_path, header):
    special = ["-0", "+0.000", "-0e+2", "1.234E-5", "+NaN", "-NaN", "Infinity", "+inf", "-INFINITY", " \r\n1.00 \t"]
    expected = [str(j) for j in range(78)]
    expected[:len(special)] = special
    for name in ("Init Fwd Win Byts", "Init Bwd Win Byts"):
        expected[MODEL_FEATURE_COLUMNS.index(name)] = "-1"
    rows = [[expected[MODEL_FEATURE_COLUMNS.index(name)] if name in MODEL_FEATURE_COLUMNS else "private" for name in header]]
    source, output = files(tmp_path, header=header, rows=rows)
    compat.prepare(source, output)
    assert decoded(output.read_bytes())[1] == expected
    parsed = csv_replay.parse_csv(io.BytesIO(output.read_bytes())).rows[0]
    assert np.signbit(parsed[0]) and np.signbit(parsed[2])
    assert np.isnan(parsed[4:6]).all() and np.isposinf(parsed[6:8]).all() and np.isneginf(parsed[8])
    for name in ("Init Fwd Win Byts", "Init Bwd Win Byts"):
        assert parsed[MODEL_FEATURE_COLUMNS.index(name)] == -1


def test_determinism_hashes_and_unverified_declaration(tmp_path):
    source, output = files(tmp_path)
    a = compat.prepare(source, output, exporter="Declared Product", exporter_version="1.0")
    second = tmp_path / "second.csv"
    b = compat.prepare(source, second, exporter="Declared Product", exporter_version="1.0")
    assert a == b and output.read_bytes() == second.read_bytes()
    assert compat.sidecar_path(output).read_bytes() == compat.sidecar_path(second).read_bytes()
    assert a["source"] == compat._byte_info(source.read_bytes())
    assert a["output"] == compat._byte_info(output.read_bytes())  # identical replay upload digest
    assert a["exporter_declaration"]["verification"] == "user-declared-unverified"
    assert len(compat.sidecar_path(output).read_bytes()) < compat.SIDECAR_LIMIT


@pytest.mark.parametrize("header", [
    CSV_COLUMNS[::-1], CSV_COLUMNS_EXTENDED[::-1], MODEL_FEATURE_COLUMNS,
    CSV_COLUMNS[:-1], (*CSV_COLUMNS, "extra"), (*CSV_COLUMNS[:-1], CSV_COLUMNS[0]),
    tuple("CWR Flag Count" if n == "CWE Flag Count" else n for n in CSV_COLUMNS),
    ("dst Port", *CSV_COLUMNS[1:]), (" Dst Port", *CSV_COLUMNS[1:]),
    ("Dst Port ", *CSV_COLUMNS[1:]), (), ("unknown",),
])
def test_unknown_profiles_are_rejected_without_publication(tmp_path, header):
    source, output = files(tmp_path, header=header, rows=[])
    with pytest.raises(compat.CompatibilityError):
        compat.preview(source)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("value", ["", " ", "N/A", "unknown", "=1+2", "1_000", "1e999", "3.5e38", "-3.5e38", "1\"2"])
def test_numeric_rejection_uses_production_parser_without_leaking_metadata(tmp_path, value):
    rows = source_rows(CSV_COLUMNS, 1)
    rows[0][0] = value
    source, output = files(tmp_path, rows=rows)
    with pytest.raises(compat.CompatibilityError, match="C2A NON_NUMERIC_FEATURE") as error:
        compat.prepare(source, output)
    assert "private" not in str(error.value)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("payload", [
    b"", encoded(CSV_COLUMNS), encoded(CSV_COLUMNS) + b"\n",
    encoded(CSV_COLUMNS, [CSV_COLUMNS]), encoded(CSV_COLUMNS, [CSV_COLUMNS_EXTENDED]),
    encoded(CSV_COLUMNS, [["0"] * 79]), encoded(CSV_COLUMNS, [["0"] * 81]),
    encoded(CSV_COLUMNS) + b'"unterminated', encoded(CSV_COLUMNS) + b'"1"x,2\n',
    encoded(CSV_COLUMNS) + b"\xff\n",
    encoded(CSV_COLUMNS, [["0"] * 80]).replace(b"0,0,0,", b'0,0,meta"bad,', 1),
])
def test_bad_records_fail_closed(tmp_path, payload):
    source, output = files(tmp_path, payload=payload)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


def test_preview_is_header_only_not_whole_file_validation(tmp_path):
    source, output = files(tmp_path, payload=encoded(CSV_COLUMNS) + b"bad-row\n")
    assert compat.preview(source)["further_record_validation_required"]
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


def test_actual_source_size_boundary(tmp_path):
    header = encoded(CSV_COLUMNS)
    row = ["0"] * 80
    row[2] = "x" * (csv_replay.MAX_BYTES - len(header) - len(encoded((), [row])) + 2)
    payload = encoded(CSV_COLUMNS, [row])
    assert len(payload) == csv_replay.MAX_BYTES
    source, output = files(tmp_path, payload=payload)
    compat.prepare(source, output)
    output.unlink()
    compat.sidecar_path(output).unlink()
    source.write_bytes(payload + b"\n")
    with pytest.raises(compat.CompatibilityError, match="limit"):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


def test_actual_output_size_overflow(tmp_path):
    payload = encoded(CSV_COLUMNS, [["12345"] * 80] * 17_000)
    assert len(payload) < csv_replay.MAX_BYTES
    source, output = files(tmp_path, payload=payload)
    with pytest.raises(compat.CompatibilityError, match="output exceeds 10 MiB"):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


def test_actual_row_count_boundary(tmp_path):
    payload = encoded(CSV_COLUMNS, [["0"] * 80] * csv_replay.MAX_ROWS)
    source, output = files(tmp_path, payload=payload)
    assert compat.prepare(source, output)["output_record_count"] == 25_000
    output.unlink()
    compat.sidecar_path(output).unlink()
    source.write_bytes(payload + encoded((), [["0"] * 80])[1:])
    with pytest.raises(compat.CompatibilityError, match="25,000"):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("which", ["output", "sidecar", "directory", "hardlink"])
def test_existing_destinations_are_never_deleted_or_changed(tmp_path, which):
    source, output = files(tmp_path)
    destination = compat.sidecar_path(output) if which == "sidecar" else output
    if which == "directory":
        destination.mkdir()
    elif which == "hardlink":
        os.link(source, destination)
    else:
        destination.write_bytes(b"pre-existing")
    before = source.read_bytes()
    with pytest.raises(compat.CompatibilityError, match="already exists"):
        compat.prepare(source, output)
    assert source.read_bytes() == before and destination.exists()
    if which not in ("directory", "hardlink"):
        assert destination.read_bytes() == b"pre-existing"
    assert len(list(tmp_path.iterdir())) == 2


def test_aliases_and_hardlinked_input_rejected(tmp_path):
    source, output = files(tmp_path)
    with pytest.raises(compat.CompatibilityError, match="distinct"):
        compat.prepare(source, source.parent / "." / source.name)
    alias = tmp_path / "alias.csv"
    os.link(source, alias)
    with pytest.raises(compat.CompatibilityError, match="Hard-linked"):
        compat.prepare(alias, output)
    alias.unlink()
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("relative", ["output.csv", "data/out.csv", "checkpoints/out.csv", "release/out.csv", "frontend/screenshots/out.csv", "e10-private/out.csv"])
def test_protected_destinations_rejected_before_access(tmp_path, relative):
    source, _ = files(tmp_path)
    with pytest.raises(compat.CompatibilityError, match="protected"):
        compat.prepare(source, compat.ROOT / relative)
    assert_no_results(tmp_path, source)


def test_protected_source_rejected_before_read(tmp_path):
    with pytest.raises(compat.CompatibilityError, match="protected"):
        compat.preview(compat.ROOT / "data/raw/csv/prohibited.csv")


def test_unavailable_filesystem_safety_fails_before_temporary_creation(tmp_path, monkeypatch):
    source, output = files(tmp_path)
    def unsupported(*args):
        raise compat.CompatibilityError("local NTFS required")
    monkeypatch.setattr(compat, "_require_ntfs", unsupported)
    with pytest.raises(compat.CompatibilityError, match="NTFS"):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("path", [r"\\server\share\out.csv", r"\\?\C:\temp\out.csv", r"C:\temp\out.csv:ads", r"C:\temp\NUL.csv", r"C:\temp\out.csv.", r"C:\temp\out.csv "])
def test_windows_ambiguous_paths_rejected(path):
    with pytest.raises(compat.CompatibilityError):
        compat._path(path)


def test_real_junction_and_repository_alias_rejected(tmp_path):
    source, output = files(tmp_path)
    for target in (tmp_path, compat.ROOT):
        junction = tmp_path / "junction"
        _winapi.CreateJunction(str(target), str(junction))
        try:
            with pytest.raises(compat.CompatibilityError):
                compat.prepare(source, junction / output.name)
        finally:
            junction.rmdir()
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("failure", ["parse", "metadata", "flush"])
def test_temporary_processing_failure_leaves_no_result(tmp_path, monkeypatch, failure):
    source, output = files(tmp_path)
    def fail(*args, **kwargs):
        raise OSError("synthetic-private-error")
    if failure == "parse":
        monkeypatch.setattr(compat, "_validate", fail)
    elif failure == "metadata":
        monkeypatch.setattr(compat, "_manifest", fail)
    else:
        monkeypatch.setattr(compat.os, "fsync", fail)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("stage,after", [(0, False), (1, False), (0, True), (1, True)])
def test_publication_failure_rolls_back_only_owned_links(tmp_path, monkeypatch, stage, after):
    source, output = files(tmp_path)
    publish = compat._publish
    calls = 0
    def fail(temporary, destination):
        nonlocal calls
        current = calls
        calls += 1
        if current != stage or after:
            publish(temporary, destination)
        if current == stage:
            raise OSError("injected publication failure")
    monkeypatch.setattr(compat, "_publish", fail)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert_no_results(tmp_path, source)


@pytest.mark.parametrize("stage", [0, 1])
def test_concurrent_destination_creation_preserves_competitor(tmp_path, monkeypatch, stage):
    source, output = files(tmp_path)
    publish = compat._publish
    calls = 0
    competitor = None
    def race(temporary, destination):
        nonlocal calls, competitor
        if calls == stage:
            competitor = destination
            destination.write_bytes(b"other-owner")
        calls += 1
        publish(temporary, destination)
    monkeypatch.setattr(compat, "_publish", race)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output)
    assert competitor.read_bytes() == b"other-owner"
    assert set(tmp_path.iterdir()) == {source, competitor}


def test_locked_parent_source_and_temporary_cannot_be_substituted(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    source, output = files(work)
    publish = compat._publish
    attempts = []
    def attack(temporary, destination):
        for operation in (lambda: work.rename(tmp_path / "moved"),
                          lambda: source.write_bytes(b"replacement"),
                          lambda: temporary.write_bytes(b"replacement"),
                          lambda: temporary.unlink()):
            with pytest.raises(OSError):
                operation()
            attempts.append(True)
        publish(temporary, destination)
    monkeypatch.setattr(compat, "_publish", attack)
    compat.prepare(source, output)
    assert len(attempts) == 8
    compat.verify(output)


@pytest.mark.parametrize("fault", ["output", "mapping", "source_hash", "flags", "count", "extra", "order", "duplicate_key", "declared", "type", "nested_metadata"])
def test_linkage_verification_detects_changed_output_or_manifest(tmp_path, fault):
    source, output = files(tmp_path)
    compat.prepare(source, output)
    path = compat.sidecar_path(output)
    record = json.loads(path.read_bytes())
    if fault == "output":
        output.write_bytes(output.read_bytes().replace(b'"0"', b'"9"', 1))
    elif fault == "duplicate_key":
        path.write_bytes(path.read_bytes().replace(b'{', b'{"rows_removed":0,', 1))
    else:
        if fault == "mapping": record["source_to_model_columns"][0]["source_index"] = 2
        elif fault == "source_hash": record["source"]["sha256"] = "0" * 64
        elif fault == "flags": record["values_imputed"] = True
        elif fault == "count": record["output_record_count"] = 1
        elif fault == "extra": record["labels"] = ["private"]
        elif fault == "order": record["original_headers"].reverse()
        elif fault == "declared": record["exporter_declaration"]["verification"] = "authenticated"
        elif fault == "type": record["rows_removed"] = False
        else:
            record["source"]["labels"] = ["private"]
            record.pop("manifest_sha256")
            record["manifest_sha256"] = canonical_sha256(record)
        path.write_bytes(compat._json(record))
    with pytest.raises(compat.CompatibilityError, match="linkage"):
        compat.verify(output)


def test_recomputed_untrusted_manifest_is_not_exporter_authentication(tmp_path):
    source, output = files(tmp_path)
    compat.prepare(source, output)
    path = compat.sidecar_path(output)
    record = json.loads(path.read_bytes())
    record["source"]["sha256"] = "0" * 64
    record.pop("manifest_sha256")
    record["manifest_sha256"] = canonical_sha256(record)
    path.write_bytes(compat._json(record))
    # Explicit evidence of the trust limit: a self-reported source digest is not authenticated.
    assert compat.verify(output)["source"]["sha256"] == "0" * 64


@pytest.mark.parametrize("fault", ["invalid_utf8", "truncated", "nested", "oversized"])
def test_malformed_sidecars_fail_with_safe_bounded_diagnostics(tmp_path, fault):
    source, output = files(tmp_path)
    compat.prepare(source, output)
    replacements = {"invalid_utf8": b"\xff", "truncated": b"{",
                    "nested": b"[" * 2000 + b"0" + b"]" * 2000,
                    "oversized": b"x" * (compat.SIDECAR_LIMIT + 1)}
    compat.sidecar_path(output).write_bytes(replacements[fault])
    with pytest.raises(compat.CompatibilityError):
        compat.verify(output)


@pytest.mark.parametrize("declaration", ["", "\nprivate", "x" * 201])
def test_bad_exporter_declarations_rejected(tmp_path, declaration):
    source, output = files(tmp_path)
    with pytest.raises(compat.CompatibilityError):
        compat.prepare(source, output, exporter=declaration)
    assert_no_results(tmp_path, source)


def test_cli_preview_prepare_verify_and_safe_errors(tmp_path, capsys):
    source, output = files(tmp_path)
    assert main(["preview", "--source", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["further_record_validation_required"]
    assert_no_results(tmp_path, source)
    assert main(["prepare", "--source", str(source), "--output", str(output), "--exporter", "declared"]) == 0
    assert json.loads(capsys.readouterr().out)["output_record_count"] == 13
    assert main(["verify", "--output", str(output)]) == 0
    capsys.readouterr()
    assert main(["prepare", "--source", str(source), "--output", str(output)]) == 2
    assert "already exists" in capsys.readouterr().err
