"""Read-only lazy Stage D4 adapter over frozen Stage C NumPy arrays.

One selected Stage C sample ``X[i]`` is exposed as the exact transpose
``feature_x`` with shape ``(78, 5)``.  No scaling, casting, reordering,
aggregation, graph-framework construction, or full-array loading occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from src.graph import schema as d2
from src.graph import taxonomy as d1
from src.graph import validation as d3


ADAPTER_VERSION = "stage-d4-lazy-sample-adapter-v1"
APPROVED_D3_REPORT_SHA256 = (
    "40f4e23e9e0937f5182ebcdb6d6363301012e05c98a90d8b3985968bd1e605bb"
)
EXPERIMENT_DIRS: Mapping[str, str] = MappingProxyType(
    {"E1": "exp1", "E2": "exp2", "E3": "exp3"}
)
EXPERIMENT_METADATA_IDENTITIES: Mapping[str, str] = MappingProxyType(
    {
        "E1": "1_global_chronological",
        "E2": "2_per_class_chronological",
        "E3": "3_binary_chronological",
    }
)
SPLIT_STEMS: Mapping[str, str] = MappingProxyType(
    {"train": "train", "validation": "val", "test": "test"}
)


class AdapterContractError(ValueError):
    """Raised when arrays, metadata, or graph provenance violate D4."""


@dataclass(frozen=True, slots=True)
class StaticGraphMetadata:
    """Immutable D1/D2 structures shared by every logical sample."""

    feature_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_category_ids: tuple[int, ...]
    category_ids: tuple[int, ...]
    sample_node_id: int
    feature_to_category_sources: tuple[int, ...]
    feature_to_category_destinations: tuple[int, ...]
    category_to_sample_sources: tuple[int, ...]
    category_to_sample_destinations: tuple[int, ...]
    d1_taxonomy_sha256: str
    d2_schema_sha256: str
    d3_report_sha256: str


STATIC_GRAPH_METADATA = StaticGraphMetadata(
    feature_ids=d2.NODE_IDS["feature"],
    feature_names=d1.MODEL_FEATURE_COLUMNS,
    feature_category_ids=d1.FEATURE_INDEX_TO_CATEGORY_ID,
    category_ids=d2.NODE_IDS["category"],
    sample_node_id=0,
    feature_to_category_sources=d2.FEATURE_TO_CATEGORY_SOURCE_INDICES,
    feature_to_category_destinations=d2.FEATURE_TO_CATEGORY_DESTINATION_INDICES,
    category_to_sample_sources=d2.CATEGORY_TO_SAMPLE_SOURCE_INDICES,
    category_to_sample_destinations=d2.CATEGORY_TO_SAMPLE_DESTINATION_INDICES,
    d1_taxonomy_sha256=d3.APPROVED_D1_TAXONOMY_SHA256,
    d2_schema_sha256=d3.APPROVED_D2_SCHEMA_SHA256,
    d3_report_sha256=APPROVED_D3_REPORT_SHA256,
)


@dataclass(frozen=True, slots=True)
class LogicalGraphSample:
    """Framework-independent view of one frozen Stage C sample."""

    feature_x: np.ndarray
    target: np.int16
    sample_index: int
    experiment: str
    split: str
    source_artifacts: Mapping[str, str]
    static: StaticGraphMetadata

    @property
    def feature_ids(self) -> tuple[int, ...]:
        return self.static.feature_ids

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.static.feature_names

    @property
    def feature_category_ids(self) -> tuple[int, ...]:
        return self.static.feature_category_ids

    @property
    def category_ids(self) -> tuple[int, ...]:
        return self.static.category_ids

    @property
    def sample_node_id(self) -> int:
        return self.static.sample_node_id


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterContractError(f"cannot load {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterContractError(f"{description} must contain a JSON object: {path}")
    return payload


def validate_graph_provenance(
    d1_manifest_path: Path,
    d2_manifest_path: Path,
    d3_report_path: Path,
) -> Mapping[str, str]:
    """Validate actual D1/D2/D3 artifacts before any Stage C array is opened."""

    try:
        d1_manifest = d2.load_validated_d1_manifest(d1_manifest_path)
        d2_manifest = _load_json(d2_manifest_path, "D2 schema manifest")
        current_d3 = d3.validate_manifest(d2_manifest, d1_manifest)
        saved_d3 = _load_json(d3_report_path, "D3 validation report")
        d3.report_bytes(saved_d3)
    except Exception as exc:
        raise AdapterContractError(f"graph provenance validation failed: {exc}") from exc
    if saved_d3 != current_d3:
        raise AdapterContractError("graph provenance validation failed: D3 report is not current")
    if saved_d3.get("report_sha256") != APPROVED_D3_REPORT_SHA256:
        raise AdapterContractError("graph provenance validation failed: D3 report hash is not approved")
    return MappingProxyType(
        {
            "d1_taxonomy_sha256": str(d1_manifest["taxonomy_sha256"]),
            "d2_schema_sha256": str(d2_manifest["schema_sha256"]),
            "d3_report_sha256": str(saved_d3["report_sha256"]),
        }
    )


class LazyGraphDataset:
    """Lazy, read-only adapter for one frozen Stage C experiment/split pair.

    Only non-negative integer indices are accepted.  The large source arrays
    remain NumPy memory maps opened with ``mmap_mode='r'``.
    """

    def __init__(
        self,
        experiment: str,
        split: str,
        *,
        processed_root: Path | None = None,
        d1_manifest_path: Path | None = None,
        d2_manifest_path: Path | None = None,
        d3_report_path: Path | None = None,
    ) -> None:
        if experiment not in EXPERIMENT_DIRS:
            raise AdapterContractError(
                f"experiment must be one of {tuple(EXPERIMENT_DIRS)}, got {experiment!r}"
            )
        if split not in SPLIT_STEMS:
            raise AdapterContractError(
                f"split must be one of {tuple(SPLIT_STEMS)}, got {split!r}"
            )
        root = _project_root()
        self.experiment = experiment
        self.split = split
        self._processed_root = (
            Path(processed_root)
            if processed_root is not None
            else root / "data" / "processed" / "preprocessed"
        )
        self.provenance = validate_graph_provenance(
            Path(d1_manifest_path)
            if d1_manifest_path is not None
            else root / "results" / "stage_d1" / "feature_taxonomy.json",
            Path(d2_manifest_path)
            if d2_manifest_path is not None
            else root / "results" / "stage_d2" / "graph_schema.json",
            Path(d3_report_path)
            if d3_report_path is not None
            else root / "results" / "stage_d3" / "structural_validation.json",
        )
        if dict(self.provenance) != {
            "d1_taxonomy_sha256": STATIC_GRAPH_METADATA.d1_taxonomy_sha256,
            "d2_schema_sha256": STATIC_GRAPH_METADATA.d2_schema_sha256,
            "d3_report_sha256": STATIC_GRAPH_METADATA.d3_report_sha256,
        }:
            raise AdapterContractError("graph provenance does not match shared static metadata")

        directory = EXPERIMENT_DIRS[experiment]
        stem = SPLIT_STEMS[split]
        experiment_dir = self._processed_root / directory
        self.x_path = experiment_dir / f"X_{stem}.npy"
        self.y_path = experiment_dir / f"y_{stem}.npy"
        self.metadata_path = experiment_dir / "split_metadata.json"
        self.source_artifacts: Mapping[str, str] = MappingProxyType(
            {
                "x": f"data/processed/preprocessed/{directory}/X_{stem}.npy",
                "y": f"data/processed/preprocessed/{directory}/y_{stem}.npy",
                "split_metadata": (
                    f"data/processed/preprocessed/{directory}/split_metadata.json"
                ),
            }
        )
        metadata = _load_json(self.metadata_path, "Stage C split metadata")
        if metadata.get("experiment") != EXPERIMENT_METADATA_IDENTITIES[experiment]:
            raise AdapterContractError(
                f"split metadata experiment mismatch for {experiment}: "
                f"{metadata.get('experiment')!r}"
            )

        try:
            self._x = np.load(self.x_path, mmap_mode="r", allow_pickle=False)
            self._y = np.load(self.y_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise AdapterContractError(f"cannot open frozen Stage C arrays read-only: {exc}") from exc
        self._validate_arrays(metadata, stem)
        self.static = STATIC_GRAPH_METADATA

    def _validate_arrays(self, metadata: Mapping[str, Any], stem: str) -> None:
        if not isinstance(self._x, np.memmap) or not isinstance(self._y, np.memmap):
            raise AdapterContractError("Stage C arrays must be NumPy memory maps")
        if self._x.flags.writeable or self._y.flags.writeable:
            raise AdapterContractError("Stage C arrays must be opened read-only")
        if self._x.dtype != np.dtype(np.float32):
            raise AdapterContractError(f"X dtype must be float32, got {self._x.dtype}")
        if self._x.ndim != 3 or self._x.shape[1:] != (5, 78):
            raise AdapterContractError(f"X shape must be (N, 5, 78), got {self._x.shape}")
        if self._y.dtype != np.dtype(np.int16):
            raise AdapterContractError(f"y dtype must be int16, got {self._y.dtype}")
        if self._y.ndim != 1:
            raise AdapterContractError(f"y shape must be (N,), got {self._y.shape}")
        if len(self._x) != len(self._y):
            raise AdapterContractError(
                f"X/y length mismatch: {len(self._x)} != {len(self._y)}"
            )
        metadata_count = metadata.get(f"n_{stem}")
        if metadata_count != len(self._x):
            raise AdapterContractError(
                f"split metadata count n_{stem}={metadata_count!r} "
                f"does not match arrays {len(self._x)}"
            )
        if self._x.shape[2] != len(self.static_feature_names):
            raise AdapterContractError("X feature dimension differs from D1 feature count")
        if self._x.shape[1] != d2.TEMPORAL_SEMANTICS["sequence_length"]:
            raise AdapterContractError("X temporal dimension differs from D2 temporal contract")

    @property
    def static_feature_names(self) -> tuple[str, ...]:
        return STATIC_GRAPH_METADATA.feature_names

    @property
    def x_memmap(self) -> np.memmap:
        return self._x

    @property
    def y_memmap(self) -> np.memmap:
        return self._y

    @property
    def is_lazy_read_only(self) -> bool:
        return (
            isinstance(self._x, np.memmap)
            and isinstance(self._y, np.memmap)
            and not self._x.flags.writeable
            and not self._y.flags.writeable
        )

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> LogicalGraphSample:
        if isinstance(index, bool):
            raise TypeError("sample index must be an integer, not bool")
        try:
            normalized = operator.index(index)
        except TypeError as exc:
            raise TypeError("sample index must be an integer") from exc
        if normalized < 0:
            raise IndexError("negative sample indices are not supported")
        if normalized >= len(self):
            raise IndexError(f"sample index {normalized} out of range for length {len(self)}")
        source = self._x[normalized]
        feature_x = source.transpose(1, 0)
        if feature_x.flags.writeable:
            raise AdapterContractError("adapted feature view unexpectedly became writable")
        target = self._y[normalized]
        if not isinstance(target, np.int16):
            raise AdapterContractError("target scalar did not preserve int16 dtype")
        return LogicalGraphSample(
            feature_x=feature_x,
            target=target,
            sample_index=normalized,
            experiment=self.experiment,
            split=self.split,
            source_artifacts=self.source_artifacts,
            static=self.static,
        )


def _sha256(path: Path, block_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_integrity(project_root: Path) -> dict[str, Any]:
    """Compare all Stage C artifact and raw CSV bytes with frozen evidence."""

    evidence = _load_json(
        project_root / "results" / "stage_c_full_validation.json",
        "frozen Stage C validation evidence",
    )
    processed_root = project_root / "data" / "processed" / "preprocessed"
    raw_root = project_root / "data" / "raw" / "csv"
    expected_artifacts = evidence.get("artefact_sha256", {})
    expected_raw = evidence.get("raw_file_fingerprints", {})
    actual_artifact_paths = {
        path.relative_to(processed_root).as_posix(): path
        for path in processed_root.rglob("*")
        if path.is_file()
    }
    artifact_set_matches = set(actual_artifact_paths) == set(expected_artifacts)
    artifact_matches = artifact_set_matches and all(
        _sha256(actual_artifact_paths[name]) == digest
        for name, digest in expected_artifacts.items()
    )
    actual_raw_paths = {path.name: path for path in raw_root.glob("*.csv")}
    raw_set_matches = set(actual_raw_paths) == set(expected_raw)
    raw_matches = raw_set_matches and all(
        _sha256(actual_raw_paths[name]) == details["sha256"]
        for name, details in expected_raw.items()
    )
    return {
        "stage_c_artifact_count": len(actual_artifact_paths),
        "stage_c_expected_count": len(expected_artifacts),
        "stage_c_file_set_matches": artifact_set_matches,
        "stage_c_sha256_matches": artifact_matches,
        "raw_csv_count": len(actual_raw_paths),
        "raw_csv_expected_count": len(expected_raw),
        "raw_csv_file_set_matches": raw_set_matches,
        "raw_csv_sha256_matches": raw_matches,
        "overall_status": "PASS" if artifact_matches and raw_matches else "FAIL",
    }


def build_validation_report(
    project_root: Path,
    *,
    integrity_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Spot-check every real split and build a deterministic D4 report."""

    project_root = Path(project_root)
    processed_root = project_root / "data" / "processed" / "preprocessed"
    datasets: dict[tuple[str, str], LazyGraphDataset] = {}
    split_reports: list[dict[str, Any]] = []
    all_transposes = True
    all_targets = True
    all_feature_order = True
    all_time_order = True
    all_memmaps = True
    all_finite = True

    for experiment in EXPERIMENT_DIRS:
        for split in SPLIT_STEMS:
            dataset = LazyGraphDataset(experiment, split, processed_root=processed_root)
            datasets[(experiment, split)] = dataset
            tested_indices = [0, len(dataset) // 2, len(dataset) - 1]
            for index in tested_indices:
                sample = dataset[index]
                source = dataset.x_memmap[index]
                all_transposes &= bool(np.array_equal(sample.feature_x, source.T, equal_nan=True))
                all_targets &= bool(sample.target == dataset.y_memmap[index])
                all_feature_order &= all(
                    np.array_equal(sample.feature_x[feature_id], source[:, feature_id], equal_nan=True)
                    for feature_id in range(78)
                )
                all_time_order &= all(
                    np.array_equal(sample.feature_x[:, time_id], source[time_id, :], equal_nan=True)
                    for time_id in range(5)
                )
                all_finite &= bool(np.isfinite(sample.feature_x).all())
                if sample.static is not STATIC_GRAPH_METADATA:
                    raise AdapterContractError("static topology object was not reused")
            all_memmaps &= dataset.is_lazy_read_only
            split_reports.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "x_artifact": dataset.source_artifacts["x"],
                    "y_artifact": dataset.source_artifacts["y"],
                    "metadata_artifact": dataset.source_artifacts["split_metadata"],
                    "length": len(dataset),
                    "x_shape": list(dataset.x_memmap.shape),
                    "x_dtype": str(dataset.x_memmap.dtype),
                    "y_shape": list(dataset.y_memmap.shape),
                    "y_dtype": str(dataset.y_memmap.dtype),
                    "tested_indices": tested_indices,
                    "read_only_memory_map": dataset.is_lazy_read_only,
                }
            )

    e1_e3_equal = True
    for split in SPLIT_STEMS:
        e1_dataset = datasets[("E1", split)]
        e3_dataset = datasets[("E3", split)]
        for index in (0, len(e1_dataset) // 2, len(e1_dataset) - 1):
            e1_e3_equal &= bool(
                np.array_equal(
                    e1_dataset[index].feature_x,
                    e3_dataset[index].feature_x,
                    equal_nan=True,
                )
            )
    e2_independent = all(
        datasets[("E2", split)].x_path != datasets[("E1", split)].x_path
        and datasets[("E2", split)].x_path.parent.name == "exp2"
        for split in SPLIT_STEMS
    )
    topology_reused = len({id(dataset.static) for dataset in datasets.values()}) == 1
    integrity = dict(integrity_result) if integrity_result is not None else verify_frozen_integrity(project_root)
    provenance = dict(next(iter(datasets.values())).provenance)
    overall = all(
        (
            all_transposes,
            all_targets,
            all_feature_order,
            all_time_order,
            all_memmaps,
            all_finite,
            e1_e3_equal,
            e2_independent,
            topology_reused,
            integrity.get("overall_status") == "PASS",
        )
    )
    report: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "provenance": provenance,
        "experiment_splits": split_reports,
        "transpose_validation": {
            "definition": "feature_x[f,t] == StageC_X[i,t,f]",
            "all_tested_samples_exact": all_transposes,
        },
        "target_validation": {
            "definition": "target == StageC_y[i]",
            "all_tested_targets_exact_int16": all_targets,
        },
        "feature_order_validation": {
            "frozen_order_preserved": all_feature_order,
            "feature_count": 78,
        },
        "temporal_order_validation": {
            "position_0": "earliest flow in frozen T=5 window",
            "position_4": "fifth/final flow and target source",
            "all_tested_time_positions_preserved": all_time_order,
        },
        "topology_validation": {
            "shared_static_object": topology_reused,
            "feature_to_category_edges": 78,
            "category_to_sample_edges": 8,
            "total_structural_edges": 86,
        },
        "e1_e3_consistency": {
            "selected_feature_x_equal": e1_e3_equal,
            "targets_not_required_equal": True,
        },
        "e2_independence": {
            "uses_exp2_artifacts_not_exp1": e2_independent,
        },
        "resource_validation": {
            "numpy_memory_map": all_memmaps,
            "read_only": all_memmaps,
            "full_array_scan_performed_by_adapter": False,
            "selected_samples_finite": all_finite,
        },
        "stage_c_integrity": integrity,
        "overall_status": "PASS" if overall else "FAIL",
    }
    report["report_sha256"] = d2.canonical_sha256(report)
    if not overall:
        raise AdapterContractError("D4 validation report contains a failed contract check")
    return report


def report_bytes(report: Mapping[str, Any]) -> bytes:
    supplied_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if not isinstance(supplied_hash, str) or d2.canonical_sha256(unhashed) != supplied_hash:
        raise AdapterContractError("D4 report SHA-256 does not match canonical content")
    return (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def write_validation_report(
    project_root: Path, output_path: Path, *, verify_integrity: bool = True
) -> dict[str, Any]:
    integrity = verify_frozen_integrity(project_root) if verify_integrity else None
    report = build_validation_report(project_root, integrity_result=integrity)
    data = report_bytes(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the read-only Stage D4 adapter")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_d4/lazy_adapter_validation.json"),
    )
    args = parser.parse_args(argv)
    report = write_validation_report(args.project_root, args.output)
    print(f"wrote {args.output}")
    print(f"overall_status={report['overall_status']}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
