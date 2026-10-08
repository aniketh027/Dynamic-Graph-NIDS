from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from backend.app.core.config import Settings
from backend.app.core.errors import ApplicationError
from backend.app.main import create_app
from backend.app.schemas.contracts import ErrorResponse
from backend.app.services import artifacts
from backend.app.services.inference import FrozenInferenceService
from src.preprocessing.schema import MODEL_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def deployment():
    # Compact frozen evidence and checkpoint byte hashes only; no prediction/data.
    return artifacts.discover(ROOT)


def service_for(deployment, loader=lambda *_: (object(), object())):
    return FrozenInferenceService(ROOT, discoverer=lambda _: deployment, loader=loader)


def test_all_five_get_routes_and_canonical_metadata(deployment):
    with TestClient(create_app(Settings(), service_for(deployment))) as client:
        assert client.get("/api/v1/health").json() == {"status": "ok", "service": "graphnids-backend"}
        state = client.get("/api/v1/status").json()
        assert state["model_adapter_state"] == "ready" and state["model_loaded"]
        assert not state["inference_ready"] and state["file_replay_supported"] and not state["live_capture_supported"]
        model = client.get("/api/v1/model").json()
        assert model["deployment_seed"] == 2026
        assert model["checkpoint_id"] == "e10-htgnn-e2-lr1e-3-s2026-b256-0a1122c0ee42"
        assert model["checkpoint_sha256"] == "ef7976f4beb58813a596453d65b448cb1a09f3998ac5865b667d51795dcf135f"
        assert (model["model_input_features"], model["retained_feature_nodes"], model["removed_feature_nodes"],
                model["category_nodes"], model["total_graph_nodes_unpruned"], model["total_graph_nodes_pruned"],
                model["edges_unpruned"], model["edges_pruned"], model["temporal_window"], model["class_count"]) == (78,61,17,8,87,70,86,69,5,15)
        assert model["model_parameters_unchanged"] and not model["selection_uses_test_evidence"]
        schema = client.get("/api/v1/schema").json()
        assert schema["feature_names"] == list(MODEL_FEATURE_COLUMNS) and schema["shape"] == [5,78]
        assert not schema["arbitrary_61_column_input_supported"] and schema["application_input_available"]
        caps = client.get("/api/v1/capabilities").json()
        assert caps["model_inference_adapter"] == "ready" and caps["demo_frontend"] == "frontend-only"
        assert caps["csv_file_replay_inference"] and caps["application_prediction_endpoint"]
        assert "checkpoint_path" not in model and "C:" not in str(model)


def test_loading_disabled_reports_unloaded_but_authenticated_metadata(deployment):
    def fail(*_):
        pytest.fail("Disabled model initialization called the loader")
    with TestClient(create_app(Settings(load_model=False), service_for(deployment, fail))) as client:
        assert client.get("/api/v1/status").json()["model_adapter_state"] == "uninitialized"
        assert client.get("/api/v1/model").json()["deployment_seed"] == 2026


@pytest.mark.parametrize("failed_at", ["discovery", "load"])
def test_failed_startup_keeps_health_status_and_schema_available(deployment, failed_at):
    def fail(*_):
        raise RuntimeError("private C:/secret/checkpoint.pt debug details")
    service = service_for(deployment, fail if failed_at == "load" else lambda *_: (object(),object()))
    if failed_at == "discovery":
        service._discover = fail
    with TestClient(create_app(Settings(), service)) as client:
        assert client.get("/api/v1/health").status_code == 200
        status = client.get("/api/v1/status")
        assert status.json()["model_adapter_state"] == "error" and not status.json()["model_loaded"]
        assert "private" not in status.text and "secret" not in status.text
        assert client.get("/api/v1/schema").status_code == 200
        response = client.get("/api/v1/model")
        assert response.status_code == (503 if failed_at == "discovery" else 200)
        if failed_at == "discovery":
            assert ErrorResponse.model_validate(response.json()).error.code == "BACKEND_NOT_READY"


@pytest.mark.parametrize("method,path,expected", [("get","/api/v1/missing",404), ("post","/api/v1/health",405), ("post","/api/v1/predict",404)])
def test_unimplemented_operations_have_typed_errors(deployment, method, path, expected):
    with TestClient(create_app(Settings(load_model=False), service_for(deployment))) as client:
        response = getattr(client,method)(path)
        assert response.status_code == expected
        assert ErrorResponse.model_validate(response.json()).error.code == "CAPABILITY_NOT_IMPLEMENTED"


def test_invalid_request_and_unexpected_errors_are_sanitized(deployment):
    app = create_app(Settings(load_model=False), service_for(deployment))
    @app.get("/test/validation")
    def synthetic_validation(count: int):
        return count
    @app.get("/test/failure")
    def synthetic_failure():
        raise RuntimeError("secret stack content")
    with TestClient(app, raise_server_exceptions=False) as client:
        invalid = client.get("/test/validation?count=private")
        assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "INVALID_REQUEST"
        failed = client.get("/test/failure")
        assert failed.status_code == 500 and "secret" not in failed.text
        assert ErrorResponse.model_validate(failed.json()).error.code == "INTERNAL_ERROR"


@pytest.mark.parametrize("origin,allowed", [("http://localhost:5173",True),("http://127.0.0.1:5173",True),("https://untrusted.example",False)])
def test_cors_is_explicit_and_allows_replay_posts(deployment, origin, allowed):
    with TestClient(create_app(Settings(load_model=False), service_for(deployment))) as client:
        response = client.options("/api/v1/health",headers={"Origin":origin,"Access-Control-Request-Method":"GET"})
        assert (response.headers.get("access-control-allow-origin") == origin) is allowed
        assert response.status_code == (200 if allowed else 400)
        blocked = client.options("/api/v1/health",headers={"Origin":origin,"Access-Control-Request-Method":"POST"})
        assert blocked.status_code == (200 if allowed else 400)


@pytest.mark.parametrize("origin", ["*","http://localhost:5173/path","http://user:password@localhost:5173","file:///tmp"])
def test_invalid_cors_configuration_rejected(origin):
    with pytest.raises(ValueError):
        Settings(cors_origins=(origin,))


@pytest.mark.parametrize("scores,seed", [([.8,.9,.7],2026),([.9,.9,.7],2025),([.7,.9,.9],2026),([.9,.9,.9],2025)])
def test_selection_uses_pruned_validation_and_lowest_seed_exact_ties(scores, seed):
    assert artifacts.choose_seed([{"seed":s,"macro_f1":v} for s,v in zip(artifacts.SEEDS,scores)]) == seed


@pytest.mark.parametrize("fault", ["missing","duplicate","nan","out_of_range","boolean"])
def test_malformed_seed_selection_fails_closed(fault):
    scores=[{"seed":2025,"macro_f1":.8},{"seed":2026,"macro_f1":.9},{"seed":2027,"macro_f1":.7}]
    if fault == "missing": scores.pop()
    elif fault == "duplicate": scores[2]["seed"]=2026
    elif fault == "nan": scores[0]["macro_f1"]=float("nan")
    elif fault == "out_of_range": scores[0]["macro_f1"]=1.1
    else: scores[0]["macro_f1"]=True
    with pytest.raises(artifacts.ArtifactError): artifacts.choose_seed(scores)


def test_frozen_integrity_rejects_changed_evidence_and_checkpoint(monkeypatch):
    original = artifacts.physical_sha256
    for suffix in ("validation_sweep.json", "best.pt"):
        monkeypatch.setattr(artifacts,"physical_sha256",lambda p: "0"*64 if str(p).endswith(suffix) else original(p))
        with pytest.raises(artifacts.ArtifactError): artifacts.discover(ROOT)


def test_changed_canonical_hash_is_rejected():
    with pytest.raises(artifacts.ArtifactError): artifacts.canonical({"x":1,"hash":"0"*64},"hash")


@pytest.mark.parametrize("path", ["../best.pt", "C:/outside/best.pt"])
def test_path_escape_rejected(path):
    with pytest.raises(artifacts.ArtifactError): artifacts.confined_path(ROOT,path)


@pytest.mark.parametrize("fault", ["61_features","four_flows","reordered","nan","boolean","huge_integer","non_numeric","invalid_row"])
def test_invalid_numeric_window_contract_rejected(deployment, fault):
    service=service_for(deployment); names=list(MODEL_FEATURE_COLUMNS); values=[[0.]*78 for _ in range(5)]
    if fault == "61_features": values=[[0.]*61 for _ in range(5)]
    elif fault == "four_flows": values.pop()
    elif fault == "reordered": names.reverse()
    elif fault == "nan": values[0][0]=float("nan")
    elif fault == "boolean": values[0][0]=True
    elif fault == "huge_integer": values[0][0]=10**1000
    elif fault == "non_numeric": values[0][0]="0.0"
    else: values[0]=None
    with pytest.raises(ApplicationError,match="78-feature|78 finite") as error: service.predict_window(names,values)
    assert error.value.code == "SCHEMA_MISMATCH"


def test_valid_window_never_enables_application_prediction(deployment):
    service=service_for(deployment); names=list(MODEL_FEATURE_COLUMNS); values=[[0.]*78 for _ in range(5)]
    with pytest.raises(ApplicationError) as error: service.predict_window(names,values)
    assert error.value.code == "MODEL_NOT_LOADED"
    service.initialize(load_model=True)
    with pytest.raises(ApplicationError) as error: service.predict_window(names,values)
    assert error.value.code == "CAPABILITY_NOT_IMPLEMENTED"
    service.close(); assert service.status().model_adapter_state == "uninitialized"


def test_discovery_does_not_read_test_evidence(monkeypatch):
    original=artifacts.bound_json; seen=[]
    def record(root,path,digest):
        seen.append(path)
        return original(root,path,digest)
    monkeypatch.setattr(artifacts,"bound_json",record)
    assert artifacts.discover(ROOT).record["configuration"]["seed"] == 2026
    assert all("stage_e11" not in p and "stage_e12b" not in p and not p.startswith("data/") for p in seen)
