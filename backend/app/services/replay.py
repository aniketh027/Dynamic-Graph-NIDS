"""Application-only orchestration; frozen inference and transforms remain authoritative."""
from dataclasses import asdict
from pathlib import Path
import threading
import uuid

from backend.app.core.errors import ApplicationError
from backend.app.schemas.replay import (
    API_VERSION, AnalysisResponse, Prediction, Preprocessing, ReplayGraph,
    ReplayModel, ReplayProvenance, ValidationResponse,
)
from backend.app.services.csv_replay import (
    BATCH_SIZE, TRANSFORM_ID, TRANSFORM_SHA256, WINDOW, load_transform, parse_csv, transform_batch,
)
from src.graph.schema import canonical_sha256
from src.preprocessing.labels import BENIGN_LABEL, INT_TO_LABEL

# One active job across app instances within the supported single-worker process.
ANALYSIS_LOCK = threading.Lock()


class ReplayService:
    def __init__(self, root: Path, *, enabled: bool = True):
        self.root, self.enabled = root, enabled
        self._adapter = None
        self.lock = ANALYSIS_LOCK

    @property
    def ready(self):
        return self._adapter is not None

    def close(self):
        self._adapter = None

    def _get_adapter(self):
        if not self.enabled:
            raise ApplicationError("MODEL_NOT_READY", "Model loading is disabled in server settings.", 503)
        if self._adapter is None:
            try:
                from backend.app.ml.application_inference import ApplicationInferenceAdapter
                from backend.app.ml.external_windows import TransformIdentity
                self._adapter = ApplicationInferenceAdapter(
                    self.root, transform_state=TransformIdentity(TRANSFORM_ID, TRANSFORM_SHA256))
            except Exception as exc:
                raise ApplicationError("MODEL_NOT_READY", "Selected checkpoint or frozen policy authentication failed.", 503) from exc
        return self._adapter

    def process(self, source, *, filename: str, size: int, sha256: str, analyze: bool):
        parsed = parse_csv(source)
        if analyze and parsed.window_count == 0:
            raise ApplicationError("INSUFFICIENT_FLOWS", "Analysis requires at least five flow records.", 422)
        state = load_transform(self.root)
        analysis_id = f"{API_VERSION}-{uuid.uuid4()}"
        if not analyze:
            return ValidationResponse(
                validation_id=analysis_id, filename=filename, size_bytes=size, sha256=sha256,
                flow_row_count=len(parsed.rows), columns_reordered=parsed.columns_reordered,
                complete_window_count=parsed.window_count, trailing_flow_count=len(parsed.rows) % WINDOW,
                transform_state_sha256=TRANSFORM_SHA256)

        import torch
        adapter = None
        predictions = []
        counts = {label: 0 for label in INT_TO_LABEL.values()}
        # Each batch owns only its transformed windows and one PyG graph batch.
        for start in range(0, parsed.window_count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, parsed.window_count)
            raw = parsed.rows[start * WINDOW:stop * WINDOW].reshape(-1, WINDOW, 78)
            transformed = transform_batch(raw, state)
            if adapter is None:
                adapter = self._get_adapter()
            try:
                owned = adapter.windows(transformed, source_id=f"sha256:{sha256}", analysis_id=analysis_id,
                                        replay_order_semantics="csv-row-order", max_windows=BATCH_SIZE)
                with torch.inference_mode():
                    logits = adapter.infer(owned, max_windows=BATCH_SIZE)
                    if logits.shape != (stop - start, len(INT_TO_LABEL)) or not torch.isfinite(logits).all():
                        raise ValueError("invalid E2 logits")
                    classes = logits.argmax(dim=1)
                    scores = torch.softmax(logits, dim=1).gather(1, classes[:, None]).flatten()
                    for offset, (class_id, score) in enumerate(zip(classes.tolist(), scores.tolist())):
                        label = INT_TO_LABEL[class_id]
                        k = start + offset
                        predictions.append(Prediction(window_index=k, target_flow_index=k * WINDOW + WINDOW - 1,
                                                      prediction_class_id=class_id, prediction_label=label,
                                                      model_score=score, is_attack=label != BENIGN_LABEL))
                        counts[label] += 1
            except Exception as exc:
                self._adapter = None  # Fail closed; the next request must reauthenticate.
                raise ApplicationError("INFERENCE_FAILED", "Frozen application inference failed; no partial predictions returned.", 503) from exc
        identity = asdict(adapter.identity)
        transform_identity = identity.pop("transform_state")
        # Retain the adapter's identities, but give uploaded row order its own explicit semantics.
        for key in ("window_semantics", "target_semantics", "temporal_window_size"):
            identity.pop(key)
        provenance = ReplayProvenance(
            **identity, analysis_id=analysis_id, uploaded_file_sha256=sha256,
            transform_state_id=transform_identity["state_id"], transform_state_sha256=transform_identity["sha256"])
        policy = adapter.policy.record()
        graph = policy["pruned_graph"]
        return AnalysisResponse(
            analysis_id=analysis_id, source_filename=filename, source_sha256=sha256, source_size_bytes=size,
            input_flow_count=len(parsed.rows), complete_window_count=parsed.window_count,
            trailing_flow_count=len(parsed.rows) % WINDOW, columns_reordered=parsed.columns_reordered,
            benign_prediction_count=counts[BENIGN_LABEL], intrusion_prediction_count=len(predictions) - counts[BENIGN_LABEL],
            prediction_class_counts=counts, predictions=predictions,
            model=ReplayModel(checkpoint_id=adapter.identity.checkpoint_id, checkpoint_sha256=adapter.identity.checkpoint_sha256),
            preprocessing=Preprocessing(transform_state_id=TRANSFORM_ID, transform_state_sha256=TRANSFORM_SHA256),
            graph=ReplayGraph(input_feature_nodes=policy["original_graph"]["feature_nodes"],
                              retained_feature_nodes=graph["feature_nodes"], category_nodes=graph["category_nodes"],
                              total_nodes_per_graph=graph["nodes"], edges_per_graph=graph["edges"],
                              parameter_count=sum(p.numel() for p in adapter.model.parameters())),
            provenance=provenance, provenance_sha256=canonical_sha256(provenance.model_dump(mode="json")))
