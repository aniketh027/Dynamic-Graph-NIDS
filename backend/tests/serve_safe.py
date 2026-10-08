"""Local HTTP smoke server with C1 guards and the real frozen loader."""
from backend.tests.safety import install_guard


def main() -> None:
    violations = install_guard()
    import torch
    import uvicorn

    def no_training(*args, **kwargs):
        violations.append("training/optimizer/backward attempted")
        raise PermissionError("C1 forbids training, optimizers and backward passes")

    torch.optim.Optimizer.__init__ = no_training
    torch.Tensor.backward = no_training
    torch.autograd.backward = no_training
    try:
        uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, log_level="info")
    finally:
        print("C1 HTTP smoke guard violations:", len(violations))


if __name__ == "__main__":
    main()
