"""Run external-adapter tests with data/write/training/experiment guards."""
import os
from pathlib import Path
import sys

from backend.tests.safety import ROOT, install_guard

REGRESSIONS = (
    "backend/tests/test_foundation.py", "tests/test_graph_schema.py",
    "tests/test_graph_batching.py", "tests/test_pyg_adapter.py",
    "tests/test_e2_provenance_cache.py", "tests/test_feature_encoder.py",
    "tests/test_hierarchical_encoder.py", "tests/test_htgnn_classifier.py",
    *(f"tests/test_stage_e12_pruning.py::{name}" for name in (
        "test_zero_pruning_is_exact_original_logits_and_weights",
        "test_policy_counts_subset_and_all_category_connectivity",
        "test_invalid_ratios_fail", "test_invalid_scores_fail",
        "test_manual_policy_cannot_remove_categories_add_nodes_or_violate_quotas",
        "test_compaction_maps_original_edges_without_new_edges_or_cross_graphs",
        "test_pruned_execution_deterministic_batched_isolation_and_skips_removed_encoder_rows",
        "test_malformed_inputs_fail_before_gat",
        "test_attention_shape_mapping_and_eval_normalization_installed_pyg",
        "test_inference_checkpoint_loader_no_optimizer_state_bytes_unchanged",
        "test_inference_loader_rejects_corrupted_binding",
        "test_deterministic_context_restores_runtime_settings",
    )),
    "-k", "not real and not backward and not zero_gradient and not e1_e3 "
    "and not bounded_heterodata_cuda_transfer and not all_nine_pairs and not train_dropout "
    "and not saved_d6_report_is_deterministic_and_complete "
    "and not task_contract_protection_does_not_freeze_numerical_parameters",
)


def main() -> int:
    replay = "--replay" in sys.argv[1:]
    arguments = [arg for arg in sys.argv[1:] if arg != "--replay"]
    scaler = ROOT / "data/processed/preprocessed/exp2/scaler_state.json"
    violations = install_guard(allowed_read_files=(scaler,) if replay else ())

    def hook(event, args):
        if event != "open" or isinstance(args[0], (int, type(None))):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if not path.is_relative_to(ROOT):
            return
        relative = path.relative_to(ROOT).as_posix()
        compact_provenance = {
            "logs/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/manifest.json",
            "logs/stage_e12/e10-official-20261001-v1/e12a-20261006-v1/status.json",
        }
        if ((relative.startswith("logs/") and relative not in compact_provenance)
                or relative.split("/")[0].startswith(("e10-", "run_e10_"))
                or path.suffix.lower() in {".csv", ".npy", ".npz"}
                or (path.name == "scaler_state.json" and not (replay and path == scaler))):
            violations.append(f"application validation prohibited read: {relative}")
            raise PermissionError(f"Application validation rejected non-permitted research access: {relative}")

    sys.addaudithook(hook)
    import torch
    import pytest
    from src.pruning import stage_e12
    from src.training import harness
    from src.preprocessing import transforms

    def forbidden(*args, **kwargs):
        violations.append("training/optimizer/backward/ranking/evaluation/benchmark attempted")
        raise PermissionError("Application adapter validation is synthetic inference only")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    torch.autograd.grad = forbidden
    transforms.fit_transforms = forbidden
    transforms.fit_transforms_indexed = forbidden
    transforms._fit_from_column_loader = forbidden
    harness.run_training = forbidden
    harness.create_optimizer = forbidden
    for name in ("collect_scores", "evaluate_validation", "benchmark", "run_phase_a", "select_candidate"):
        setattr(stage_e12, name, forbidden)
    requested = REGRESSIONS if arguments == ["--regressions"] else (
        arguments or ["backend/tests/test_replay.py" if replay else "backend/tests/test_external_application.py"])
    code = pytest.main(["-q", "-p", "no:cacheprovider", *requested])
    print("Application adapter guard violations:", len(violations))
    return code if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
