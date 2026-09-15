"""Stage D7 bounded real construction and mutated-fixture tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.graph import adapter as d4
from src.graph import batching as d6
from src.graph import construction_sanity as d7


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "results" / "stage_d7" / "construction_sanity_validation.json"


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> d4.LazyGraphDataset:
    directory = tmp_path / "exp1"
    directory.mkdir(parents=True)
    values = np.arange(9 * 5 * 78, dtype=np.float32).reshape(9, 5, 78)
    targets = np.arange(9, dtype=np.int16)
    np.save(directory / "X_train.npy", values, allow_pickle=False)
    np.save(directory / "y_train.npy", targets, allow_pickle=False)
    (directory / "split_metadata.json").write_text(
        json.dumps({"experiment": "1_global_chronological", "n_train": 9}),
        encoding="utf-8",
    )
    return d4.LazyGraphDataset("E1", "train", processed_root=tmp_path)


def _fixture_batch(
    dataset: d4.LazyGraphDataset, indices: list[int]
) -> d6.LogicalGraphBatch:
    return d6.build_logical_batch(
        dataset, indices, _allow_noncanonical_fixture=True
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def test_deterministic_selection_rule() -> None:
    assert d7.deterministic_sanity_indices(100) == (0, 25, 50, 75, 99)
    assert d7.deterministic_sanity_indices(101) == (0, 25, 50, 75, 100)


def test_selection_rejects_too_short_and_invalid_length() -> None:
    with pytest.raises(d7.ConstructionSanityError, match="five unique"):
        d7.deterministic_sanity_indices(4)
    with pytest.raises(TypeError, match="integer"):
        d7.deterministic_sanity_indices(True)


def test_valid_sample_contract_and_reproducible_fingerprint(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    sample = synthetic_dataset[4]
    d7.validate_sample_contract(synthetic_dataset, 4, sample)
    first = d7.logical_sample_fingerprint(sample)
    second = d7.logical_sample_fingerprint(synthetic_dataset[4])
    assert first == second
    assert len(first) == 64


def test_valid_batch_contract_and_reproducible_fingerprint(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    indices = [8, 1, 5]
    batch = _fixture_batch(synthetic_dataset, indices)
    d7.validate_batch_contract(synthetic_dataset, indices, batch)
    assert d7.logical_batch_fingerprint(batch) == d7.logical_batch_fingerprint(
        _fixture_batch(synthetic_dataset, indices)
    )


@pytest.mark.parametrize("field", tuple(d7.APPROVED_PROVENANCE))
def test_corrupted_d1_through_d6_provenance_is_rejected(field: str) -> None:
    provenance = {**d7.APPROVED_PROVENANCE, field: "0" * 64}
    with pytest.raises(d7.ConstructionSanityError, match=field):
        d7.validate_stage_provenance(provenance)


def test_wrong_feature_shape_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = synthetic_dataset[3]
    mutated = replace(sample, feature_x=sample.feature_x[:-1])
    with pytest.raises(d7.ConstructionSanityError, match="wrong feature shape"):
        d7.validate_sample_contract(synthetic_dataset, 3, mutated)


def test_wrong_feature_dtype_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = synthetic_dataset[3]
    mutated = replace(sample, feature_x=sample.feature_x.astype(np.float64))
    with pytest.raises(d7.ConstructionSanityError, match="wrong feature dtype"):
        d7.validate_sample_contract(synthetic_dataset, 3, mutated)


def test_wrong_target_dtype_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = replace(synthetic_dataset[3], target=np.int32(3))
    with pytest.raises(d7.ConstructionSanityError, match="wrong target dtype"):
        d7.validate_sample_contract(synthetic_dataset, 3, sample)


def test_feature_value_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = synthetic_dataset[3]
    feature_x = sample.feature_x.copy()
    feature_x[0, 0] += 1
    with pytest.raises(d7.ConstructionSanityError, match="feature value mismatch"):
        d7.validate_sample_contract(
            synthetic_dataset, 3, replace(sample, feature_x=feature_x)
        )


def test_feature_order_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = synthetic_dataset[3]
    with pytest.raises(d7.ConstructionSanityError, match="feature order mismatch"):
        d7.validate_sample_contract(
            synthetic_dataset, 3, replace(sample, feature_x=sample.feature_x[::-1])
        )


def test_temporal_order_reversal_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = synthetic_dataset[3]
    with pytest.raises(d7.ConstructionSanityError, match="temporal order reversal"):
        d7.validate_sample_contract(
            synthetic_dataset, 3, replace(sample, feature_x=sample.feature_x[:, ::-1])
        )


def test_target_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = replace(synthetic_dataset[3], target=np.int16(4))
    with pytest.raises(d7.ConstructionSanityError, match="target mismatch"):
        d7.validate_sample_contract(synthetic_dataset, 3, sample)


def test_sample_experiment_mismatch_is_rejected(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    sample = replace(synthetic_dataset[3], experiment="E2")
    with pytest.raises(d7.ConstructionSanityError, match="experiment mismatch"):
        d7.validate_sample_contract(synthetic_dataset, 3, sample)


def test_sample_split_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    sample = replace(synthetic_dataset[3], split="test")
    with pytest.raises(d7.ConstructionSanityError, match="split mismatch"):
        d7.validate_sample_contract(synthetic_dataset, 3, sample)


def test_wrong_edge_count_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    indices = [1, 7]
    batch = _fixture_batch(synthetic_dataset, indices)
    mutated = replace(
        batch,
        feature_to_category_sources=_readonly(
            batch.feature_to_category_sources[:-1].copy()
        ),
    )
    with pytest.raises(d7.ConstructionSanityError, match="wrong feature-to-category"):
        d7.validate_batch_contract(synthetic_dataset, indices, mutated)


def test_cross_sample_edge_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    indices = [1, 7]
    batch = _fixture_batch(synthetic_dataset, indices)
    destinations = batch.feature_to_category_destinations.copy()
    destinations[0] = 8
    mutated = replace(
        batch, feature_to_category_destinations=_readonly(destinations)
    )
    with pytest.raises(d7.ConstructionSanityError, match="cross-sample"):
        d7.validate_batch_contract(synthetic_dataset, indices, mutated)


def test_batch_experiment_mismatch_is_rejected(
    synthetic_dataset: d4.LazyGraphDataset,
) -> None:
    indices = [1, 7]
    batch = replace(_fixture_batch(synthetic_dataset, indices), experiment="E2")
    with pytest.raises(d7.ConstructionSanityError, match="batch experiment"):
        d7.validate_batch_contract(synthetic_dataset, indices, batch)


def test_batch_split_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    indices = [1, 7]
    batch = replace(_fixture_batch(synthetic_dataset, indices), split="test")
    with pytest.raises(d7.ConstructionSanityError, match="batch split"):
        d7.validate_batch_contract(synthetic_dataset, indices, batch)


def test_batch_feature_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    indices = [1, 7]
    batch = _fixture_batch(synthetic_dataset, indices)
    feature_x = batch.feature_x.copy()
    feature_x[1, 0, 0] += 1
    mutated = replace(batch, feature_x=_readonly(feature_x))
    with pytest.raises(d7.ConstructionSanityError, match="batch feature value"):
        d7.validate_batch_contract(synthetic_dataset, indices, mutated)


def test_batch_target_mismatch_is_rejected(synthetic_dataset: d4.LazyGraphDataset) -> None:
    indices = [1, 7]
    batch = _fixture_batch(synthetic_dataset, indices)
    targets = batch.targets.copy()
    targets[1] = 1
    mutated = replace(batch, targets=_readonly(targets))
    with pytest.raises(d7.ConstructionSanityError, match="batch target mismatch"):
        d7.validate_batch_contract(synthetic_dataset, indices, mutated)


def test_nondeterministic_fingerprint_fixture_is_rejected() -> None:
    first = d7.SanityRunResult(
        records=({"sample_fingerprint": "a"},),
        topology_identity="topology",
        e1_e3_features_equal=True,
        e1_e3_targets_convert=True,
        e2_provenance_valid=True,
        no_model_input_leakage=True,
    )
    second = replace(first, records=({"sample_fingerprint": "b"},))
    with pytest.raises(d7.ConstructionSanityError, match="nondeterministic"):
        d7.validate_repeatability(first, second)


def test_real_report_covers_all_nine_pairs_and_is_deterministic() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    regenerated = d7.run_construction_sanity(
        ROOT, integrity_result=saved["stage_c_integrity"]
    )
    assert regenerated == saved
    assert d7.report_bytes(regenerated) == REPORT_PATH.read_bytes()
    assert len(saved["experiment_split_results"]) == 9
    assert saved["selected_sample_count_per_run"] == 45
    assert saved["tested_batch_sizes"] == [1, 2, 5]
    assert saved["overall_status"] == "PASS"


def test_report_rejects_corrupted_hash() -> None:
    saved = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    saved["report_sha256"] = "0" * 64
    with pytest.raises(d7.ConstructionSanityError, match="SHA-256"):
        d7.validate_report(saved)
