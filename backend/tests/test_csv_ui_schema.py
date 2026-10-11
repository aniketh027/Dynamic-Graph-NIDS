"""Synthetic read-only schema/route tests. No frozen artifacts or inference."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

from backend.app.core.config import Settings
from backend.app.main import create_app
from backend.app.services.csv_schema import csv_descriptor, CONTRACT_VERSION
from backend.app.services import csv_replay, inference
from src.graph.schema import canonical_sha256
from src.preprocessing.schema import CSV_COLUMNS, CSV_COLUMNS_EXTENDED, MODEL_FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


def generator():
    spec = importlib.util.spec_from_file_location("c3b_generator", ROOT / "frontend/scripts/generate-csv-schema.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile,columns", [("standard_80", CSV_COLUMNS), ("extended_84", CSV_COLUMNS_EXTENDED)])
def test_descriptor_uses_authority(profile, columns):
    descriptor = csv_descriptor()
    assert descriptor["contract_version"] == CONTRACT_VERSION
    assert descriptor["canonical_columns"] == list(MODEL_FEATURE_COLUMNS)
    layout = descriptor["profiles"][profile]
    assert layout["columns"] == list(columns)
    assert [columns[index] for index in layout["projection_indices"]] == list(MODEL_FEATURE_COLUMNS)
    assert layout["removed_metadata_fields"] == [name for name in columns if name not in MODEL_FEATURE_COLUMNS]
    assert descriptor["feature_order_sha256"] == canonical_sha256(MODEL_FEATURE_COLUMNS)
    digest = descriptor.pop("descriptor_sha256")
    assert digest == canonical_sha256(descriptor)


def test_numeric_limits_whitespace_and_no_global_mutation():
    before = csv.field_size_limit()
    descriptor = csv_descriptor()
    assert csv.field_size_limit() == before
    numeric, limits = descriptor["numeric"], descriptor["limits"]
    assert numeric["pattern"] == csv_replay.NUMBER.pattern
    assert float(numeric["float32_max_finite"]) == csv_replay.FLOAT32_MAX
    assert numeric["strip_code_points"] == [n for n in range(0x110000) if chr(n).isspace()]
    assert 0x1c in numeric["strip_code_points"] and 0xfeff not in numeric["strip_code_points"]
    assert limits["retained_field_characters"] == before == 131072
    assert limits["source_bytes"] == limits["output_bytes"] == csv_replay.MAX_BYTES
    assert limits["flow_records"] == csv_replay.MAX_ROWS == 25000
    assert limits["prediction_windows"] == 5000
    assert limits["window"] == csv_replay.WINDOW == 5
    assert limits["inference_batch_windows"] == csv_replay.BATCH_SIZE == 256


def test_generation_determinism_check_and_drift(tmp_path, capsys):
    module = generator()
    expected = module.render()
    assert expected == module.render() == module.TARGET.read_bytes()
    assert module.main(["--check"]) == 0
    module.TARGET = tmp_path / "manifest.ts"
    assert module.main(["--check"]) == 1
    assert not module.TARGET.exists()
    module.TARGET.write_bytes(expected.replace(b'"standard_80"', b'"standard_81"'))
    before = module.TARGET.read_bytes()
    assert module.main(["--check"]) == 1
    assert module.TARGET.read_bytes() == before
    assert "drift" in capsys.readouterr().err
    assert module.main([]) == 0
    assert module.TARGET.read_bytes() == expected
    assert module.main(["--check"]) == 0


def test_schema_route_additive_and_no_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Schema/health must not initialize or read model state")
    service = inference.FrozenInferenceService(ROOT, discoverer=forbidden, loader=forbidden)
    monkeypatch.setattr(service, "initialize", lambda **kwargs: None)
    app = create_app(Settings(load_model=False), service)
    old = inference.schema_metadata().model_dump(mode="json")
    with TestClient(app) as client:
        value = client.get("/api/v1/schema").json()
        assert value["csv_compatibility"] == csv_descriptor()
        assert value == old
        assert value["feature_names"] == list(MODEL_FEATURE_COLUMNS)
        assert value["shape"] == [5, 78]
        assert value["prediction_target"] == "fifth/final flow"
        assert client.get("/api/v1/health").json() == {"status": "ok", "service": "graphnids-backend"}
        status = client.get("/api/v1/status").json()
        assert status["model_loaded"] is False and status["inference_ready"] is False
        capabilities = client.get("/api/v1/capabilities").json()
        assert capabilities["csv_file_replay_inference"] is True
        assert capabilities["raw_pcap"] is False and capabilities["live_packet_capture"] is False
        paths = app.openapi()["paths"]
        assert "post" in paths["/api/v1/replay/validate"]
        assert "post" in paths["/api/v1/replay/analyze"]
        assert not any("prepare" in path for path in paths)
    assert not any(name.startswith(("torch", "src.training", "src.pruning", "backend.app.ml")) for name in sys.modules)


def test_descriptor_new_objects_cannot_change_authority():
    first = csv_descriptor()
    first["canonical_columns"][0] = "changed"
    first["profiles"]["standard_80"]["projection_indices"][0] = 2
    first["numeric"]["strip_code_points"].clear()
    assert csv_descriptor()["canonical_columns"] == list(MODEL_FEATURE_COLUMNS)
    assert csv_descriptor()["profiles"]["standard_80"]["projection_indices"][0] == 0
    assert csv_descriptor()["numeric"]["strip_code_points"]
    assert json.dumps(csv_descriptor(), sort_keys=True) == json.dumps(csv_descriptor(), sort_keys=True)
