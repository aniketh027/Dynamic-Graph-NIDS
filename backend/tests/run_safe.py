"""Run only explicitly selected C1-safe tests under read/write/training guards."""
import sys

from backend.tests.safety import install_guard


def main() -> int:
    violations = install_guard()
    import torch
    import pytest

    def no_training(*args, **kwargs):
        violations.append("training/optimizer/backward attempted")
        raise PermissionError("C1 forbids training, optimizers and backward passes")

    torch.optim.Optimizer.__init__ = no_training
    torch.Tensor.backward = no_training
    torch.autograd.backward = no_training
    code = pytest.main(["-q", "-p", "no:cacheprovider", *(sys.argv[1:] or ["backend/tests/test_foundation.py"])])
    print("C1 guard violations:", len(violations))
    return code if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
