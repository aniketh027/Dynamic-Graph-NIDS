from pathlib import Path
import hashlib,json,os,sys,tempfile
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["GIT_OPTIONAL_LOCKS"] = "0"
root=Path(__file__).resolve().parents[4]
scratch=Path(tempfile.mkdtemp(prefix="e12b-authority-validation-"))
sys.path.insert(0,str(root))
blocked=[]
def audit(event,args):
    if event != "open" or not isinstance(args[0],(str,bytes,os.PathLike)): return
    path=Path(os.path.abspath(os.fsdecode(args[0]))).resolve()
    if root not in path.parents and root != path: return
    mode,flags=args[1:3]
    writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if writing or (path.suffix==".npy" and root/"data" in path.parents):
        blocked.append({"event":event,"path":str(path),"writing":bool(writing)})
        raise AssertionError("real checkout writes/predictive array opens prohibited during synthetic verification")
sys.addaudithook(audit)
import torch,pytest
from src.pruning import stage_e12b as b
from src.training import harness as e7
execution_attempts=[]
def forbidden(*args,**kwargs):
    execution_attempts.append("backward/optimizer/training")
    raise AssertionError("training or optimizer step prohibited")
torch.Tensor.backward=forbidden
torch.autograd.backward=forbidden
torch.optim.AdamW.step=forbidden
e7.run_training=forbidden
e7.train_one_epoch=forbidden
class Results:
    def __init__(self): self.passed=self.failed=self.skipped=self.deselected=0
    def pytest_runtest_logreport(self,report):
        if report.when=="call":
            self.passed+=report.passed; self.failed+=report.failed; self.skipped+=report.skipped
        elif report.failed: self.failed+=1
    def pytest_deselected(self,items): self.deselected+=len(items)
result=Results()
args=["-q","-p","no:cacheprovider","tests/test_stage_e10_full_training.py","tests/test_stage_e11_test_evaluation.py","tests/test_stage_e11_failure_recovery.py","tests/test_stage_e12_pruning.py","tests/test_stage_e12b.py","tests/test_stage_e9_batch_size_pilot.py","-k","not test_run_candidate_short_real_cuda_timing_and_memory_path"]
exit_code=pytest.main(args,plugins=[result])
with b.file_guard(root) as access:
    attestation=b.verify_authority_reconciliation(root)
    e9=b.a.e10._validate_frozen_e9_evidence(root)
    assert e9["e9_source_sha256"]["PROJECT_HANDOFF.md"]==b.HISTORICAL_HANDOFF_SHA256
    assert not access.opened_test and not access.attempts
    assert not (root/"logs/stage_e12b").exists() and not (root/"results/stage_e12b").exists()
summary={"pytest_args":args,"exit_code":exit_code,"passed":result.passed,"failed":result.failed,"skipped":result.skipped,"deselected":result.deselected,"real_repository_write_or_predictive_open_attempts":blocked,"training_backward_or_optimizer_step_attempts":execution_attempts,"live_authority_validation":"PASS","semantic_e9_observation_validation":"PASS","attestation_canonical_sha256":attestation["attestation_sha256"],"test_arrays_opened":[],"e12b_execution_namespaces_exist":False,"full_project_suite_run":False,"full_suite_exclusion_reason":"Contains real predictive accesses and training tests outside owner scope"}
(scratch/"allowed_validation.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf-8")
print("VALIDATION_RESULTS",str(scratch/"allowed_validation.json"))
print("GUARDED_VALIDATION",json.dumps(summary,sort_keys=True))
raise SystemExit(exit_code)
