"""C3A exact-layout, text-preserving offline preparation. No model operations.

Publication requires Windows and an owner-controlled local NTFS directory.
Directory handles deny write/delete sharing; regular-file handles deny writes
and deletion. Hard links provide exclusive, no-replace publication. The sidecar
is published first, and the validated CSV last. This is not a crash transaction
or a sandbox against hostile same-user/native code.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
import csv
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import uuid

from backend.app.core.errors import ApplicationError
from backend.app.services import csv_replay
from src.graph.schema import canonical_json_bytes, canonical_sha256
from src.preprocessing.schema import (
    CSV_COLUMNS, CSV_COLUMNS_EXTENDED, MODEL_FEATURE_COLUMNS,
    VARIANT_STANDARD, VARIANT_EXTENDED,
)

CONTRACT_VERSION = "graphnids-c3a-exact-csv-v1"
ROOT = Path(__file__).resolve().parents[3]
SIDECAR_LIMIT = 64 * 1024
ORDER_SEMANTICS = "zero-based source record i equals output record i; csv-row-order-preserved"


class CompatibilityError(ValueError):
    """Safe diagnostics; never include removed cells or native exception paths."""


def _profile(header):
    if tuple(header) == CSV_COLUMNS:
        return VARIANT_STANDARD
    if tuple(header) == CSV_COLUMNS_EXTENDED:
        return VARIANT_EXTENDED
    if len(set(header)) != len(header):
        raise CompatibilityError("Duplicate source headers; use the exact documented 80/84 layout.")
    raise CompatibilityError("Unrecognized source schema; require exact ordered 80/84 headers, with no aliases or extra fields.")


def _path(value):
    original = Path(os.fspath(value))
    path = Path(os.path.abspath(os.fspath(value)))
    # Refuse device namespaces, UNC/network paths, alternate data streams,
    # Win32 trailing-dot/space aliases and reserved device basenames.
    if os.name != "nt":
        raise CompatibilityError("Safe file preparation currently requires Windows local NTFS paths.")
    if not re.fullmatch(r"[A-Za-z]:\\", path.anchor):
        raise CompatibilityError("Use a local drive path, not a network or device namespace.")
    # Validate before GetFullPathNameW/abspath can normalize a trailing dot/space.
    for part in original.parts[1:] if original.anchor else original.parts:
        if (part.endswith((".", " ")) or ":" in part
                or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise CompatibilityError("Unsafe Windows path component; choose an ordinary file path.")
    if path.is_relative_to(ROOT) or path.resolve().is_relative_to(ROOT.resolve()):
        raise CompatibilityError("Repository and its protected data/artifact/screenshot roots cannot be used as input or output.")
    return path


def _reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _identity(info):
    return info.st_dev, info.st_ino


def _require_ntfs(anchor):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes = [wintypes.LPCWSTR]
    drive_type.restype = wintypes.UINT
    volume = kernel.GetVolumeInformationW
    volume.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
                       ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                       ctypes.POINTER(wintypes.DWORD), wintypes.LPWSTR, wintypes.DWORD]
    volume.restype = wintypes.BOOL
    filesystem = ctypes.create_unicode_buffer(32)
    if (drive_type(anchor) not in (2, 3)
            or not volume(anchor, None, 0, None, None, None, filesystem, len(filesystem))
            or filesystem.value != "NTFS"):
        raise CompatibilityError("Safe publication requires a local NTFS drive; other filesystems and mapped network drives are unsupported.")


@contextmanager
def _windows_handle(path, *, directory=False, create=False):
    """Open the actual object without following a reparse point; deny mutations."""
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    open_file = kernel.CreateFileW
    open_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                          wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    open_file.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    query.restype = wintypes.BOOL

    class AttributeTag(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

    access = 0x80 if directory else (0xC0000000 if create else 0x80000000)
    handle = open_file(str(path), access, 1, None, 1 if create else 3,
                       0x00200000 | (0x02000000 if directory else 0), None)
    if handle == wintypes.HANDLE(-1).value:
        raise CompatibilityError("Cannot open an exclusive safe file/directory handle; check existence, permissions and concurrent users.")
    fd = None
    entered = False
    created_identity = None
    try:
        if create:
            created_identity = _identity(path.lstat())
        attrs = AttributeTag()
        if not query(handle, 9, ctypes.byref(attrs), ctypes.sizeof(attrs)):
            raise CompatibilityError("Cannot validate the opened filesystem object.")
        if (attrs.attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or bool(attrs.attributes & stat.FILE_ATTRIBUTE_DIRECTORY) != directory):
            raise CompatibilityError("Symlinks, junctions, reparse points and nonregular files are unsupported.")
        if directory:
            yield None
        else:
            fd = msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDWR if create else os.O_RDONLY))
            handle = None  # descriptor now owns the native handle
            with os.fdopen(fd, "w+b" if create else "rb") as stream:
                fd = None
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise CompatibilityError("Supply a regular CSV file.")
                entered = True
                yield stream
    finally:
        if fd is not None:
            os.close(fd)
        if handle is not None:
            close(handle)
        if create and not entered and created_identity is not None:
            _remove_owned(path, created_identity)


def _pin(stack, parent):
    """Pin every ancestor, top down, and check lexical and resolved identity."""
    _require_ntfs(parent.anchor)
    for directory in reversed((parent, *parent.parents)):
        try:
            before = directory.lstat()
            if _reparse(before) or not stat.S_ISDIR(before.st_mode):
                raise CompatibilityError("All parent directories must exist and have no reparse points.")
            stack.enter_context(_windows_handle(directory, directory=True))
            if _identity(before) != _identity(directory.lstat()):
                raise CompatibilityError("A parent directory changed while being pinned.")
        except OSError:
            raise CompatibilityError("All parent directories must already exist and be accessible.") from None


def _read(stream, limit):
    if os.fstat(stream.fileno()).st_size > limit:
        raise CompatibilityError(f"File exceeds the {limit}-byte limit; no splitting or truncation is performed.")
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise CompatibilityError(f"File exceeds the {limit}-byte limit.")
    return payload


@contextmanager
def _source(path, limit):
    with ExitStack() as stack:
        _pin(stack, path.parent)
        stream = stack.enter_context(_windows_handle(path))
        if os.fstat(stream.fileno()).st_nlink != 1:
            raise CompatibilityError("Hard-linked inputs are unsupported; supply an independent regular file.")
        yield _read(stream, limit)


@contextmanager
def _reader(payload):
    old_limit = csv.field_size_limit()
    csv.field_size_limit(csv_replay.MAX_BYTES)
    try:
        text = payload.decode("utf-8-sig", errors="strict")
        yield csv.reader(io.StringIO(text, newline=""), strict=True)
    except (UnicodeError, csv.Error):
        raise CompatibilityError("Use strict UTF-8 CSV with valid quoting and record widths.") from None
    finally:
        csv.field_size_limit(old_limit)


def _strict_quotes(payload):
    """CSV lexical syntax only, including discarded fields; no numeric parser.

    Python's strict reader permits a quote inside an unquoted field. Reject it
    and characters after a closing quote, including in removed metadata.
    """
    text = payload.decode("utf-8-sig", errors="strict")
    state = "start"
    for char in text:
        if state == "quoted":
            if char == '"':
                state = "closed"
        elif state == "closed":
            if char == '"':
                state = "quoted"
            elif char in ",\r\n":
                state = "start"
            else:
                raise CompatibilityError("Malformed CSV quoting; no characters may follow a closing quote.")
        elif char == '"':
            if state != "start":
                raise CompatibilityError("Malformed CSV quoting; quote the entire cell.")
            state = "quoted"
        elif char in ",\r\n":
            state = "start"
        else:
            state = "plain"
    if state == "quoted":
        raise CompatibilityError("Malformed CSV quoting; a quoted cell is unterminated.")


def preview(source):
    """Read-only header inspection. Never assert validity of subsequent records."""
    path = _path(source)
    with _source(path, csv_replay.MAX_BYTES) as payload, _reader(payload) as reader:
        header = next(reader, [])
        profile = _profile(header)
    return {"schema_recognized": True, "source_profile": profile,
            "source_column_count": len(header), "output_feature_count": len(MODEL_FEATURE_COLUMNS),
            "removed_metadata_fields": [name for name in header if name not in MODEL_FEATURE_COLUMNS],
            "further_record_validation_required": True, "exporter_authenticated": False}


def sidecar_path(output):
    return Path(str(output) + ".provenance.json")


def _declaration(exporter, version):
    for value in (exporter, version):
        if value is not None and (not isinstance(value, str) or not value.strip()
                                  or len(value) > 200 or not value.isprintable()):
            raise CompatibilityError("Exporter/version declarations must be nonblank printable text of at most 200 characters.")
    return {"exporter": exporter, "version": version, "verification": "user-declared-unverified"}


def _manifest(header, source_info, output_info, count, declaration):
    record = {
        "contract_version": CONTRACT_VERSION, "source_profile": _profile(header),
        "original_headers": list(header), "source": source_info, "output": output_info,
        "feature_order_sha256": canonical_sha256(MODEL_FEATURE_COLUMNS),
        "source_to_model_columns": [
            {"source_index": header.index(name), "model_index": i, "name": name}
            for i, name in enumerate(MODEL_FEATURE_COLUMNS)],
        "removed_metadata_fields": [name for name in header if name not in MODEL_FEATURE_COLUMNS],
        "source_record_count": count, "output_record_count": count,
        "record_order_semantics": ORDER_SEMANTICS, "rows_removed": 0,
        "units_changed": False, "values_imputed": False, "transforms_applied": False,
        "exporter_declaration": declaration,
    }
    return {**record, "manifest_sha256": canonical_sha256(record)}


def _byte_info(payload):
    return {"byte_count": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json(record):
    return canonical_json_bytes(record) + b"\n"


def _validate(payload):
    old_limit = csv.field_size_limit()
    csv.field_size_limit(csv_replay.MAX_BYTES)
    try:
        parsed = csv_replay.parse_csv(io.BytesIO(payload))
        if parsed.columns_reordered:
            raise CompatibilityError("Prepared output is not in canonical feature order.")
        return len(parsed.rows)
    except ApplicationError as exc:
        raise CompatibilityError(f"C2A {exc.code}: {exc.message}") from None
    finally:
        csv.field_size_limit(old_limit)


def _verify_bytes(payload, metadata):
    try:
        manifest = json.loads(metadata)
        header = manifest["original_headers"]
        _profile(header)
        source_info = manifest["source"]
        if (set(source_info) != {"byte_count", "sha256"}
                or type(source_info["byte_count"]) is not int
                or not 0 < source_info["byte_count"] <= csv_replay.MAX_BYTES
                or re.fullmatch("[0-9a-f]{64}", source_info["sha256"]) is None):
            raise ValueError("source identity")
        declaration = manifest["exporter_declaration"]
        expected = _manifest(header, source_info, _byte_info(payload), _validate(payload),
                             _declaration(declaration["exporter"], declaration["version"]))
        if metadata != _json(expected):
            raise ValueError("manifest linkage or canonical serialization")
        return expected
    except (ValueError, TypeError, KeyError, UnicodeError, AttributeError, RecursionError):
        raise CompatibilityError("Output/sidecar linkage is invalid; restore the original pair or prepare a new pair.") from None


def verify(output):
    """Verify byte linkage and manifest semantics, not exporter authenticity."""
    output = _path(output)
    sidecar = _path(sidecar_path(output))
    with _source(output, csv_replay.MAX_BYTES) as payload, _source(sidecar, SIDECAR_LIMIT) as metadata:
        return _verify_bytes(payload, metadata)


def _publish(temporary, destination):
    os.link(temporary, destination)  # exclusive create; never replace a destination


def _remove_owned(path, identity):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if not _reparse(info) and _identity(info) == identity:
        path.unlink()


def prepare(source, output, *, exporter=None, exporter_version=None):
    """Project decoded cells, validate with unmodified C2A, publish a fresh pair."""
    source, output = _path(source), _path(output)
    sidecar = _path(sidecar_path(output))
    declaration = _declaration(exporter, exporter_version)
    if source in (output, sidecar):
        raise CompatibilityError("Source and destinations must be distinct; no input is overwritten.")
    owned = []
    succeeded = False
    with ExitStack() as pins:
        _pin(pins, output.parent)
        for destination in (output, sidecar):
            if os.path.lexists(destination):
                raise CompatibilityError("Output or sidecar already exists; select a fresh destination.")
        with _source(source, csv_replay.MAX_BYTES) as payload:
            try:
                _strict_quotes(payload)
            except UnicodeError:
                raise CompatibilityError("Use strict UTF-8 CSV.") from None
            token = uuid.uuid4().hex
            csv_temp = output.parent / f".graphnids-{token}.csv.tmp"
            json_temp = output.parent / f".graphnids-{token}.json.tmp"
            try:
                with ExitStack() as handles:
                    streams = []
                    for temporary in (csv_temp, json_temp):
                        stream = handles.enter_context(_windows_handle(temporary, create=True))
                        owned.append((temporary, _identity(os.fstat(stream.fileno()))))
                        streams.append(stream)
                    projected, metadata = streams
                    count = 0
                    with _reader(payload) as reader:
                        header = next(reader, [])
                        _profile(header)
                        indices = [header.index(name) for name in MODEL_FEATURE_COLUMNS]

                        def write_row(cells):
                            row = io.StringIO(newline="")
                            # Quote every cell, including retained CR characters.
                            # Serialization is deterministic; decoded text is unchanged.
                            csv.writer(row, lineterminator="\n", quoting=csv.QUOTE_ALL).writerow(cells)
                            encoded = row.getvalue().encode("utf-8")
                            if projected.tell() + len(encoded) > csv_replay.MAX_BYTES:
                                raise CompatibilityError("Canonical output exceeds 10 MiB; no partial output is published.")
                            projected.write(encoded)

                        write_row(MODEL_FEATURE_COLUMNS)
                        for fields in reader:
                            if count == csv_replay.MAX_ROWS:
                                raise CompatibilityError("At most 25,000 flow records are accepted; no rows are dropped.")
                            if len(fields) != len(header):
                                raise CompatibilityError(f"Source record {count} has the wrong width; require {len(header)} fields.")
                            if tuple(fields) in (CSV_COLUMNS, CSV_COLUMNS_EXTENDED):
                                raise CompatibilityError(f"Embedded header at source record {count}; remove it explicitly before preparation.")
                            write_row([fields[i] for i in indices])
                            count += 1
                    projected.flush()
                    os.fsync(projected.fileno())
                    projected.seek(0)
                    canonical = _read(projected, csv_replay.MAX_BYTES)
                    if _validate(canonical) != count:
                        raise CompatibilityError("Provisional output record count differs from the source.")
                    manifest = _manifest(header, _byte_info(payload), _byte_info(canonical), count, declaration)
                    serialized = _json(manifest)
                    if len(serialized) > SIDECAR_LIMIT:
                        raise CompatibilityError("Provenance sidecar exceeds its resource bound.")
                    metadata.write(serialized)
                    metadata.flush()
                    os.fsync(metadata.fileno())
                    metadata.seek(0)
                    _verify_bytes(canonical, _read(metadata, SIDECAR_LIMIT))
                    # Keep the input and provisional-file handles locked during publication.
                    _publish(json_temp, sidecar)
                    _publish(csv_temp, output)
                    for temporary, destination in ((json_temp, sidecar), (csv_temp, output)):
                        if _identity(temporary.lstat()) != _identity(destination.lstat()):
                            raise CompatibilityError("Destination identity changed during publication.")
                succeeded = True
                return manifest
            except OSError:
                raise CompatibilityError("File processing/publication failed; check local NTFS support, free space, permissions and concurrent access.") from None
            finally:
                # Handles close before cleanup. Never remove a pre-existing or substituted file.
                if not succeeded:
                    for destination, temporary in ((output, csv_temp), (sidecar, json_temp)):
                        for path, identity in owned:
                            if path == temporary:
                                _remove_owned(destination, identity)
                for path, identity in owned:
                    _remove_owned(path, identity)
