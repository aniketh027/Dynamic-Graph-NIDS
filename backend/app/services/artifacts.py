"""Read-only discovery of the owner-approved application checkpoint.

Only E.10 and E.12A compact records and checkpoint bytes are read. No dataset
reader, experimental prepare/runner, E.11 or E.12B result is used.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from src.graph import schema as graph_schema

STUDY = "e10-official-20261001-v1"
BASE = f"results/stage_e12/{STUDY}"
FREEZE_PATH = f"{BASE}/e12a-20261006-v1-freeze/freeze_attestation.json"
SWEEP_PATH = f"{BASE}/e12a-20261006-v1/validation_sweep.json"
POLICY_PATH = f"{BASE}/e12a-20261006-v1/proposed_policy.json"
REPORT_PATH = f"results/stage_e10/{STUDY}/stage_e10_report.json"
FREEZE_PHYSICAL_SHA256 = "7f92187689993ea049cf4a364746301f1175d18ae25402e5a0518f2094032e51"
E10_PHYSICAL_SHA256 = "8ff3f89c4aac3b2948936ea794e710059a7fe441c161b8c20cae0b8e51f6639c"
SELECTION_RULE = "highest_frozen_e12a_pruned_validation_eligible_class_macro_f1_then_lowest_seed"
SEEDS = (2025, 2026, 2027)


class ArtifactError(ValueError):
    """Malformed, changed or incompatible frozen application provenance."""


def confined_path(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute() or ".." in requested.parts:
        raise ArtifactError("Artifact path must be repository-relative")
    path = (root / requested).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ArtifactError("Artifact path escaped the repository")
    return path


def physical_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound_json(root: Path, relative: str, digest: str) -> dict:
    # Compact provenance only; reject unexpectedly large replacements before JSON allocation.
    with confined_path(root, relative).open("rb") as stream:
        raw = stream.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ArtifactError("Compact provenance artifact exceeds the size limit")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ArtifactError(f"Frozen artifact bytes changed: {relative}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ArtifactError("Artifact must contain a JSON object")
    return value


def canonical(record: dict, field: str) -> None:
    if graph_schema.canonical_sha256({k: v for k, v in record.items() if k != field}) != record.get(field):
        raise ArtifactError("Artifact canonical hash mismatch")


def choose_seed(results: list[dict]) -> int:
    """Apply the owner's application-only rule to persisted scores."""
    if (len(results) != 3 or sorted(r["seed"] for r in results) != list(SEEDS)
            or any(type(r["seed"]) is not int or type(r["macro_f1"]) not in {int, float}
                   or not math.isfinite(r["macro_f1"]) or not 0 <= r["macro_f1"] <= 1
                   for r in results)):
        raise ArtifactError("Exactly three finite frozen E2 seed scores are required")
    return min(results, key=lambda r: (-r["macro_f1"], r["seed"]))["seed"]


@dataclass(frozen=True)
class Deployment:
    record: dict
    freeze: dict
    scores: dict[str, float]


def discover(root: Path) -> Deployment:
    freeze = bound_json(root, FREEZE_PATH, FREEZE_PHYSICAL_SHA256)
    canonical(freeze, "attestation_sha256")
    if (freeze["status"] != "FROZEN" or freeze["review_status"] != "PASS"
            or freeze["owner_approved_freeze"] is not True or freeze["study"] != STUDY):
        raise ArtifactError("Policy is not the reviewed frozen E.12A policy")
    for relative, digest in freeze["e12a_artifact_physical_sha256"].items():
        if physical_sha256(confined_path(root, relative)) != digest:
            raise ArtifactError("Original E.12A artifact changed")
    for relative, digest in freeze["e12a_source_physical_sha256"].items():
        if physical_sha256(confined_path(root, relative)) != digest:
            raise ArtifactError("Original E.12A implementation changed")
    policy = bound_json(root, POLICY_PATH, freeze["policy_physical_sha256"])
    canonical(policy, "policy_sha256")
    if policy["policy_sha256"] != freeze["policy_canonical_sha256"] or policy["policy"] != freeze["policy_record"]:
        raise ArtifactError("Policy/attestation binding mismatch")
    sweep = bound_json(root, SWEEP_PATH, freeze["e12a_artifact_physical_sha256"][SWEEP_PATH])
    candidates = [r for r in sweep["rows"] if r["ratio"] == freeze["selected_ratio"]
                  and r["policy"] == freeze["policy_record"]]
    if len(candidates) != 1:
        raise ArtifactError("Frozen-policy validation evidence is missing or ambiguous")
    results = candidates[0]["seed_results"]
    selected_seed = choose_seed(results)
    report = bound_json(root, REPORT_PATH, E10_PHYSICAL_SHA256)
    canonical(report, "report_sha256")
    if report["overall_status"] != "PASS" or report["official_study_id"] != STUDY:
        raise ArtifactError("Official E.10 report is incomplete or mismatched")
    checkpoints = freeze["e2_checkpoints"]
    if [r["seed"] for r in checkpoints] != list(SEEDS):
        raise ArtifactError("Frozen E2 checkpoint coverage mismatch")
    records = []
    for checkpoint in checkpoints:
        matches = [r for r in report["final_validation_selected_checkpoints"]
                   if r["candidate_id"] == checkpoint["candidate_id"]]
        if len(matches) != 1:
            raise ArtifactError("Official checkpoint membership mismatch")
        record = matches[0]
        config = record["configuration"]
        metadata = record["checkpoint_provenance"]
        if (config["task"] != "E2" or config["model_family"] != "htgnn"
                or config["seed"] != checkpoint["seed"] or metadata != checkpoint["checkpoint_provenance"]
                or record["artifacts"]["checkpoint"] != checkpoint["checkpoint_path"]
                or record["artifacts"]["checkpoint_sha256"] != checkpoint["checkpoint_physical_sha256"]
                or record["checkpoint_provenance_verified"] is not True):
            raise ArtifactError("Task/config/checkpoint/policy provenance mismatch")
        if physical_sha256(confined_path(root, checkpoint["checkpoint_path"])) != checkpoint["checkpoint_physical_sha256"]:
            raise ArtifactError("Frozen checkpoint bytes changed")
        score = next(r for r in results if r["seed"] == checkpoint["seed"])
        if (score["class_codes"] != metadata["selection_class_codes"]
                or score["examples"] != sum(map(sum, score["confusion_matrix"]))
                or score["examples"] <= 0):
            raise ArtifactError("Incomplete validation support or eligibility mismatch")
        records.append(record)
    if len({r["examples"] for r in results}) != 1 or len({r["sample_order_sha256"] for r in results}) != 1:
        raise ArtifactError("Validation membership/order differs across seeds")
    selected = next(r for r in records if r["configuration"]["seed"] == selected_seed)
    return Deployment(deepcopy(selected), deepcopy(freeze), {str(r["seed"]): r["macro_f1"] for r in results})
