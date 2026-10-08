"""C2B local browser QA server using the unchanged C2A synthetic-only guard.

Run from the repository root with .venv Python -B. No official samples.
The only data-area read exception is C2A's exact frozen E2 scaler state.
"""
import sys
import pytest
from backend.tests import run_application_safe


def serve(_arguments):
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    pytest.main = serve
    sys.argv = [sys.argv[0], "--replay"]
    raise SystemExit(run_application_safe.main())
