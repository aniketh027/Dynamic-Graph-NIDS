"""C2A synthetic CSV/HTTP validation only; no official samples or scientific evaluation."""
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path
import threading
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient
import numpy as np
import pytest
import torch

from backend.app.api import replay as routes
from backend.app.core.config import Settings
from backend.app.core.errors import ApplicationError
from backend.app.main import create_app
from backend.app.ml.application_inference import ApplicationInferenceAdapter, CHECKPOINT_ID, CHECKPOINT_SHA256
from backend.app.ml.external_windows import TransformIdentity
from backend.app.schemas.replay import AnalysisResponse, ValidationResponse
from backend.app.services import csv_replay as csv_service
from backend.app.services.inference import FrozenInferenceService
from backend.app.services.replay import ReplayService
from src.graph.schema import canonical_sha256
from src.preprocessing.labels import INT_TO_LABEL, BENIGN_LABEL
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS, SENTINEL_NEGATIVE_COLUMNS
from src.preprocessing.transforms import apply_transforms

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"Content-Type": "text/csv", "X-Filename": "synthetic.csv"}


def csv_bytes(n=10, *, names=MODEL_FEATURE_COLUMNS, values=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(names)
    if values is None:
        values = [[str((i * 78 + j) / 1000) for j in range(len(names))] for i in range(n)]
    writer.writerows(values)
    return stream.getvalue().encode("utf-8")


def app_for(*, enabled=False):
    # C1 startup uses no real loader; replay model loading is independently exercised below.
    service = FrozenInferenceService(ROOT, discoverer=lambda _: None, loader=lambda *_: (object(), object()))
    return create_app(Settings(load_model=enabled), service)


@pytest.fixture
def client():
    with TestClient(app_for()) as client:
        yield client


@pytest.fixture(scope="module")
def adapter():
    return ApplicationInferenceAdapter(ROOT, transform_state=TransformIdentity(
        csv_service.TRANSFORM_ID, csv_service.TRANSFORM_SHA256))


def post(client, body, *, analyze=False, headers=None):
    return client.post("/api/v1/replay/" + ("analyze" if analyze else "validate"),
                       content=body, headers=headers or HEADERS)


@pytest.mark.parametrize("n", [1, 4, 5, 6, 7, 8, 9, 10])
def test_validate_row_counts_sha_no_model(client, monkeypatch, n):
    def forbidden(*args, **kwargs):
        pytest.fail("validate must not load/run a model or transform rows")
    monkeypatch.setattr(ReplayService, "_get_adapter", forbidden)
    monkeypatch.setattr(ApplicationInferenceAdapter, "__init__", forbidden)
    monkeypatch.setattr(csv_service, "apply_transforms", forbidden)
    body = csv_bytes(n)
    response = post(client, body)
    assert response.status_code == 200, response.text
    result = ValidationResponse.model_validate(response.json())
    assert result.flow_row_count == n and result.complete_window_count == n // 5
    assert result.trailing_flow_count == n % 5 and result.size_bytes == len(body)
    assert result.sha256 == hashlib.sha256(body).hexdigest()
    assert result.canonical_feature_count == 78 and result.temporal_window == 5
    assert result.transform_state_sha256 == csv_service.TRANSFORM_SHA256
    assert result.replay_order == "csv-row-order" and not result.columns_reordered
    assert not result.model_inference_performed and not client.app.state.replay.ready
    assert "rows" not in response.json() and "features" not in response.json()


def test_shuffled_columns_are_reordered_by_name(client):
    original = np.arange(10 * 78, dtype=np.float32).reshape(10, 78)
    names = MODEL_FEATURE_COLUMNS[::-1]
    body = csv_bytes(names=names, values=original[:, ::-1])
    parsed = csv_service.parse_csv(io.BytesIO(body))
    assert parsed.columns_reordered
    np.testing.assert_array_equal(parsed.rows, original)
    assert post(client, body).json()["columns_reordered"]


@pytest.mark.parametrize("names,code", [
    (MODEL_FEATURE_COLUMNS[:-1], "MISSING_COLUMN"),
    (MODEL_FEATURE_COLUMNS[:61], "MISSING_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, "unknown"), "UNEXPECTED_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, "Timestamp"), "UNEXPECTED_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, "Label"), "UNEXPECTED_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, "Src IP"), "UNEXPECTED_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, "Src Port"), "UNEXPECTED_COLUMN"),
    ((*MODEL_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS[0]), "DUPLICATE_COLUMN"),
    (tuple(f"f{i}" for i in range(78)), "UNEXPECTED_COLUMN"),
])
def test_rejected_headers_before_inference(client, monkeypatch, names, code):
    monkeypatch.setattr(ReplayService, "_get_adapter", lambda *_: pytest.fail("inference after schema error"))
    for analyze in (False, True):
        response = post(client, csv_bytes(5, names=names), analyze=analyze)
        assert response.status_code == 422 and response.json()["error"]["code"] == code


@pytest.mark.parametrize("body,code", [
    (b"", "EMPTY_FILE"), (b"\n", "EMPTY_FILE"),
    (csv_bytes(0), "EMPTY_FILE"),
    (csv_bytes(0) + b'"unterminated', "CSV_PARSE_ERROR"),
    (csv_bytes(0) + b'"1"x,2\n', "CSV_PARSE_ERROR"),
    (csv_bytes(0) + b"1,2\n", "CSV_PARSE_ERROR"),
    (csv_bytes(0) + b"\xff\n", "CSV_PARSE_ERROR"),
    (csv_bytes(0) + b"\n", "CSV_PARSE_ERROR"),
    (b"PK\x03\x04archive", "UNEXPECTED_COLUMN"),
])
def test_empty_malformed_and_archive_content(client, body, code):
    response = post(client, body)
    assert response.status_code == 422 and response.json()["error"]["code"] == code


@pytest.mark.parametrize("value", ["abc", "unknown", "N/A", "", " ", "=1+2", "1_000", "1e999", "3.5e38"])
def test_strict_numeric_rejects_text_and_overflow(client, value):
    values = [["0"] * 78 for _ in range(5)]
    values[0][0] = value
    response = post(client, csv_bytes(values=values))
    assert response.status_code == 422 and response.json()["error"]["code"] == "NON_NUMERIC_FEATURE"


@pytest.mark.parametrize("header,value", [
    ("X-Filename", "data.zip"), ("X-Filename", "../data.csv"), ("X-Filename", "C:\\data.csv"),
    ("X-Filename", ""), ("Content-Type", "application/zip"),
    ("Content-Type", "multipart/form-data"), ("Content-Encoding", "gzip"),
])
def test_transport_rejects_paths_archives_and_wrong_type(client, header, value):
    response = post(client, csv_bytes(5), headers={**HEADERS, header: value})
    assert response.status_code == 415 and response.json()["error"]["code"] == "INVALID_FILE_TYPE"


def test_bom_quoted_numeric_and_dst_port(client):
    assert "Dst Port" in MODEL_FEATURE_COLUMNS
    body = b"\xef\xbb\xbf" + csv_bytes(5)
    assert post(client, body).status_code == 200


@pytest.mark.parametrize("n", [1, 4])
def test_analysis_requires_five_flows(client, n):
    response = post(client, csv_bytes(n), analyze=True)
    assert response.status_code == 422 and response.json()["error"]["code"] == "INSUFFICIENT_FLOWS"


def test_row_limit_actual_boundary(client, monkeypatch):
    body = csv_bytes(0) + (b",".join([b"0"] * 78) + b"\n") * 25_000
    assert len(body) < csv_service.MAX_BYTES
    assert post(client, body).json()["flow_row_count"] == 25_000
    monkeypatch.setattr(ReplayService, "_get_adapter", lambda *_: pytest.fail("model on oversized input"))
    response = post(client, body + b",".join([b"0"] * 78) + b"\n", analyze=True)
    assert response.status_code == 413 and response.json()["error"]["code"] == "TOO_MANY_ROWS"


def test_size_limit_early_and_streamed(client, monkeypatch):
    monkeypatch.setattr(ReplayService, "process", lambda *a, **k: pytest.fail("parse/inference on oversized upload"))
    body = b"x" * (csv_service.MAX_BYTES + 1)
    response = post(client, body, analyze=True)
    assert response.status_code == 413 and response.json()["error"]["code"] == "FILE_TOO_LARGE"
    # Generator omits Content-Length; actual streamed bytes enforce the same cap.
    response = post(client, iter([body[:1024], body[1024:]]), analyze=True)
    assert response.status_code == 413 and response.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert not client.app.state.replay.lock.locked()


def test_nonfinite_and_sentinels_follow_exact_frozen_transform(client):
    state = csv_service.load_transform(ROOT)
    raw = np.zeros((1, 5, 78), dtype=np.float32)
    raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0] = np.nan, np.inf, -np.inf
    sentinels = [MODEL_FEATURE_COLUMNS.index(name) for name in SENTINEL_NEGATIVE_COLUMNS]
    raw[:, :, sentinels] = -1
    raw[0, 3, 0] = -1
    body = csv_bytes(values=raw.reshape(5, 78))
    assert post(client, body).status_code == 200
    parsed = csv_service.parse_csv(io.BytesIO(body)).rows.reshape(1, 5, 78)
    result = csv_service.transform_batch(parsed, state)
    np.testing.assert_array_equal(result, apply_transforms(raw, state))
    for row, statistic in enumerate((state.train_median, state.train_max_finite, state.train_min_finite)):
        assert result[0, row, 0] == np.float32((statistic[0] - state.center[0]) / state.scale[0])
    assert np.all(result[:, :, sentinels] == -1)
    ordinary = np.float32((-1 - state.center[0]) / state.scale[0])
    assert result[0, 3, 0] == ordinary and ordinary != -1
    assert result.dtype == np.float32 and np.isfinite(result).all()


@pytest.mark.parametrize("fault", ["hash", "feature_order", "missing_field", "scale", "sentinel", "fit_split"])
def test_transform_authentication_failures(tmp_path, monkeypatch, fault):
    original = (ROOT / csv_service.TRANSFORM_PATH).read_bytes()
    raw = json.loads(original)
    if fault == "hash":
        payload = original + b" "
    else:
        if fault == "feature_order": raw["feature_names"].reverse()
        elif fault == "missing_field": raw.pop("center")
        elif fault == "scale": raw["scale"][0] = 0
        elif fault == "sentinel": raw["sentinel_columns"] = []
        else: raw["fit_provenance"]["fit_split"] = "test"
        payload = json.dumps(raw).encode()
        # Deliberately pass byte authentication to exercise separate semantic validation.
        monkeypatch.setattr(csv_service, "TRANSFORM_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path / csv_service.TRANSFORM_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with pytest.raises(ApplicationError) as error:
        csv_service.load_transform(tmp_path)
    assert error.value.code == "TRANSFORM_STATE_INVALID"


def test_post_transform_nonfinite_rejected(monkeypatch):
    monkeypatch.setattr(csv_service, "apply_transforms", lambda raw, _: np.full_like(raw, np.inf))
    with pytest.raises(ApplicationError) as error:
        csv_service.transform_batch(np.zeros((1, 5, 78), dtype=np.float32), None)
    assert error.value.code == "TRANSFORM_FAILED"


def test_model_disabled_is_typed(client):
    response = post(client, csv_bytes(5), analyze=True)
    assert response.status_code == 503 and response.json()["error"]["code"] == "MODEL_NOT_READY"


@pytest.mark.parametrize("n", [5, 6, 7, 8, 9, 10])
def test_real_endpoint_synthetic_smoke_repeat_and_adapter_equivalence(adapter, n):
    app = app_for(enabled=True)
    app.state.replay._adapter = adapter
    body = csv_bytes(n)
    with TestClient(app) as client:
        first = post(client, body, analyze=True)
        assert first.status_code == 200, first.text
        result = AnalysisResponse.model_validate(first.json())
        second = post(client, body, analyze=True)
        assert second.status_code == 200, second.text
        assert first.json()["predictions"] == second.json()["predictions"]
        assert result.complete_window_count == len(result.predictions) == n // 5
        assert result.trailing_flow_count == n % 5
        assert [p.target_flow_index for p in result.predictions] == list(range(4, n - n % 5, 5))
        assert result.model.checkpoint_id == CHECKPOINT_ID and result.model.checkpoint_sha256 == CHECKPOINT_SHA256
        assert result.model.deployment_seed == 2026 and result.model.task == "E2"
        assert result.model.model_score_semantics == "uncalibrated normalized model score"
        assert result.graph.model_dump() == dict(input_feature_nodes=78, retained_feature_nodes=61,
            category_nodes=8, total_nodes_per_graph=70, edges_per_graph=69, parameter_count=7183)
        p = result.provenance
        assert p.pruning_policy_sha256 == "f7a8f6aa41cdfb6ad2ec8644ef4570677b7ef172374592f84c8239d7ac0e00aa"
        assert p.source_kind == "external_application" and p.uploaded_file_sha256 == hashlib.sha256(body).hexdigest()
        assert p.analysis_id == result.analysis_id and p.api_version == "v1"
        assert p.replay_order_semantics == "csv-row-order" and p.temporal_window == 5
        assert p.transform_state_sha256 == csv_service.TRANSFORM_SHA256
        assert p.class_mapping_sha256 == canonical_sha256(INT_TO_LABEL)
        assert result.provenance_sha256 == canonical_sha256(p.model_dump(mode="json"))
        assert "split" not in type(p).model_fields and "experiment" not in type(p).model_fields
        assert set(result.prediction_class_counts) == set(INT_TO_LABEL.values())
        assert sum(result.prediction_class_counts.values()) == n // 5
        assert result.benign_prediction_count + result.intrusion_prediction_count == n // 5
        raw = csv_service.parse_csv(io.BytesIO(body)).rows[:n // 5 * 5].reshape(-1, 5, 78)
        transformed = apply_transforms(raw, csv_service.load_transform(ROOT))
        owned = adapter.windows(transformed, source_id=f"sha256:{p.uploaded_file_sha256}",
            analysis_id=result.analysis_id, replay_order_semantics="csv-row-order", max_windows=256)
        with torch.inference_mode():
            logits = adapter.infer(owned, max_windows=256)
            classes = logits.argmax(1)
            scores = logits.softmax(1).gather(1, classes[:, None]).flatten().tolist()
        assert [prediction.prediction_class_id for prediction in result.predictions] == classes.tolist()
        assert [prediction.model_score for prediction in result.predictions] == scores
        assert all(np.isfinite(prediction.model_score) and 0 <= prediction.model_score <= 1 for prediction in result.predictions)
        assert client.get("/api/v1/status").json()["inference_ready"]
        assert client.get("/api/v1/health").status_code == 200
        assert all(forbidden not in first.text for forbidden in ('"logits"', '"attention"', '"embeddings"', '"train_median"', 'C:\\'))


def test_prediction_mapping_all_classes_and_batch_bound(adapter, monkeypatch):
    app = app_for(enabled=True)
    app.state.replay._adapter = adapter
    seen = []
    original_windows = adapter.windows
    def windows(values, **kwargs):
        result = original_windows(values, **kwargs)
        seen.append(result)
        return result
    def infer(owned, **kwargs):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        b = owned.provenance.batch_size
        start = sum(item.provenance.batch_size for item in seen[:-1])
        logits = torch.zeros((b, 15))
        logits[torch.arange(b), (torch.arange(b) + start) % 15] = 2
        return logits
    monkeypatch.setattr(adapter, "windows", windows)
    monkeypatch.setattr(adapter, "infer", infer)
    with TestClient(app) as client:
        body = csv_bytes(257 * 5 + 3)
        response = post(client, body, analyze=True)
        assert response.status_code == 200, response.text
        result = response.json()
        assert [w.provenance.batch_size for w in seen] == [256, 1]
        assert result["trailing_flow_count"] == 3 and len(result["predictions"]) == 257
        for k, prediction in enumerate(result["predictions"]):
            assert prediction["window_index"] == k and prediction["target_flow_index"] == k * 5 + 4
            assert prediction["prediction_label"] == INT_TO_LABEL[k % 15]
            assert prediction["is_attack"] == (INT_TO_LABEL[k % 15] != BENIGN_LABEL)
        for item in seen:
            assert item.provenance.source_id == "sha256:" + hashlib.sha256(body).hexdigest()
            assert item.provenance.analysis_id == result["analysis_id"]
            assert item.provenance.identity.transform_state.sha256 == csv_service.TRANSFORM_SHA256
            assert item.provenance.replay_order_semantics == "csv-row-order"
            assert "split" not in asdict(item.provenance) and "experiment" not in asdict(item.provenance)


def test_busy_concurrent_request_and_validate_not_blocked(client, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = client.app.state.replay.process
    def blocked(*args, **kwargs):
        if kwargs["analyze"]:
            entered.set()
            assert release.wait(15)
            raise ApplicationError("MODEL_NOT_READY", "Synthetic blocking loader completed.", 503)
        return original(*args, **kwargs)
    monkeypatch.setattr(client.app.state.replay, "process", blocked)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(post, client, csv_bytes(5), analyze=True)
        try:
            assert entered.wait(10)
            response = post(client, csv_bytes(5), analyze=True)
            assert response.status_code == 409 and response.json()["error"]["code"] == "ANALYSIS_BUSY"
            assert post(client, csv_bytes(5)).status_code == 200
        finally:
            release.set()
        assert future.result().json()["error"]["code"] == "MODEL_NOT_READY"
    assert not client.app.state.replay.lock.locked()


@pytest.mark.parametrize("case", ["success", "parse_error", "too_large", "model_error"])
def test_spool_closed_on_all_exits(client, monkeypatch, case):
    original = routes.tempfile.SpooledTemporaryFile
    opened = []
    def tracked(*args, **kwargs):
        # Force disk spill to exercise on-close removal outside the repository.
        kwargs["max_size"] = 1
        result = original(*args, **kwargs)
        opened.append(result)
        return result
    monkeypatch.setattr(routes.tempfile, "SpooledTemporaryFile", tracked)
    body = csv_bytes(5)
    if case == "parse_error": body = csv_bytes(0) + b'"unfinished'
    if case == "too_large": body = iter([b"x" * (csv_service.MAX_BYTES + 1)])
    post(client, body, analyze=case == "model_error")
    assert len(opened) == 1 and all(item.closed for item in opened)
    assert not client.app.state.replay.lock.locked()


def test_inference_failure_is_sanitized_and_invalidates_adapter(adapter, monkeypatch):
    app = app_for(enabled=True)
    app.state.replay._adapter = adapter
    def fail(*args, **kwargs):
        raise RuntimeError("private C:/secret model state")
    monkeypatch.setattr(adapter, "infer", fail)
    with TestClient(app) as client:
        result = post(client, csv_bytes(5), analyze=True)
        assert result.status_code == 503 and result.json()["error"]["code"] == "INFERENCE_FAILED"
        assert "private" not in result.text and not app.state.replay.ready
        assert not app.state.replay.lock.locked()


def test_capabilities_are_honest(client):
    caps = client.get("/api/v1/capabilities").json()
    assert caps["csv_file_replay_inference"] and caps["strict_78_feature_application_csv"]
    assert not caps["live_packet_capture"] and not caps["raw_pcap"] and not caps["arbitrary_cicflowmeter_csv"]
    assert caps["frontend_replay_integration"] is True
    assert client.get("/api/v1/status").json()["file_replay_supported"]


def test_real_lazy_initialization_endpoint(monkeypatch):
    app = app_for(enabled=True)
    with TestClient(app) as client:
        assert not app.state.replay.ready
        response = post(client, csv_bytes(5), analyze=True)
        assert response.status_code == 200, response.text
        assert app.state.replay.ready
        assert response.json()["model"]["checkpoint_id"] == CHECKPOINT_ID


def test_upload_limits_before_body_read_and_disconnect_cleanup(client):
    import asyncio
    from starlette.requests import Request

    async def run():
        calls = 0
        async def never_read():
            nonlocal calls
            calls += 1
            raise AssertionError("Oversized declared body was read")
        scope = {"type": "http", "method": "POST", "path": "/api/v1/replay/analyze",
                 "app": client.app, "headers": [(b"content-type", b"text/csv"), (b"x-filename", b"synthetic.csv"),
                 (b"content-length", str(csv_service.MAX_BYTES + 1).encode())]}
        with pytest.raises(ApplicationError) as error:
            await routes.uploaded(Request(scope, never_read), analyze=True)
        assert error.value.code == "FILE_TOO_LARGE" and calls == 0
        scope["headers"] = scope["headers"][:2]
        async def disconnected():
            return {"type": "http.disconnect"}
        with pytest.raises(ApplicationError) as error:
            await routes.uploaded(Request(scope, disconnected), analyze=True)
        assert error.value.code == "INVALID_REQUEST"
        assert not client.app.state.replay.lock.locked()
    asyncio.run(run())


def test_guard_exact_scaler_exception_keeps_official_samples_and_writes_blocked():
    # Expected denied opens run in a separate process so the real test-run ledger
    # still detects every unplanned access. Rejection happens before any OS read.
    script = '''
from backend.tests.safety import ROOT, install_guard
scaler = ROOT / "data/processed/preprocessed/exp2/scaler_state.json"
violations = install_guard(allowed_read_files=(scaler,))
assert scaler.read_bytes()
paths = ["data/raw/csv/synthetic-prohibited.csv",
         "data/processed/preprocessed/exp2/X_train.npy",
         "data/processed/preprocessed/exp2/X_val.npy",
         "data/processed/preprocessed/exp2/X_test.npy",
         "data/processed/preprocessed/exp2/y_test.npy",
         "data/processed/preprocessed/exp1/scaler_state.json"]
for name in paths:
    try:
        (ROOT / name).open("rb")
    except PermissionError:
        pass
    else:
        raise AssertionError("guard failed")
try:
    scaler.open("wb")
except PermissionError:
    pass
else:
    raise AssertionError("write guard failed")
assert len(violations) == 7
print("7 expected guard denials; zero forbidden reads/writes")
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, check=True)
    assert "7 expected guard denials" in result.stdout


def test_shuffled_real_csv_prediction_and_trailing_rows_do_not_change_output(adapter):
    app = app_for(enabled=True)
    app.state.replay._adapter = adapter
    rows = csv_service.parse_csv(io.BytesIO(csv_bytes(5))).rows.copy()
    with TestClient(app) as client:
        original = post(client, csv_bytes(values=rows), analyze=True)
        reverse = post(client, csv_bytes(names=MODEL_FEATURE_COLUMNS[::-1], values=rows[:, ::-1]), analyze=True)
        trailing = post(client, csv_bytes(values=np.concatenate([rows, np.full((4, 78), 123, dtype=np.float32)])), analyze=True)
        assert original.status_code == reverse.status_code == trailing.status_code == 200
        assert original.json()["predictions"] == reverse.json()["predictions"] == trailing.json()["predictions"]
        assert reverse.json()["columns_reordered"] and trailing.json()["trailing_flow_count"] == 4
