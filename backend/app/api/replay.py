"""Byte-bounded raw CSV uploads, with request-scoped spooling outside the repository."""
import hashlib
from pathlib import Path
import tempfile

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from backend.app.core.errors import ApplicationError
from backend.app.schemas.contracts import ErrorResponse
from backend.app.schemas.replay import AnalysisResponse, ValidationResponse
from backend.app.services.csv_replay import MAX_BYTES

router = APIRouter(prefix="/api/v1/replay", responses={
    code: {"model": ErrorResponse} for code in (400, 409, 413, 415, 422, 503)
})
UPLOAD_DOC = {
    "requestBody": {"required": True, "content": {"text/csv": {"schema": {"type": "string", "format": "binary"}}}},
    "parameters": [{"name": "X-Filename", "in": "header", "required": True,
                    "schema": {"type": "string"}, "description": "CSV basename only; never a filesystem path."}],
}


async def uploaded(request: Request, *, analyze: bool):
    service = request.app.state.replay
    acquired = False
    try:
        if analyze:
            acquired = service.lock.acquire(blocking=False)
            if not acquired:
                raise ApplicationError("ANALYSIS_BUSY", "One analysis is already active. Retry after it completes.", 409)
        filename = request.headers.get("x-filename", "")
        if (not filename or len(filename) > 255 or any(c in filename for c in '/\\:')
                or any(ord(c) < 32 or ord(c) == 127 for c in filename)
                or not filename.lower().endswith(".csv")):
            raise ApplicationError("INVALID_FILE_TYPE", "Supply a .csv basename in X-Filename; paths and archives are unsupported.", 415)
        if (request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/csv"
                or request.headers.get("content-encoding", "identity").lower() != "identity"):
            raise ApplicationError("INVALID_FILE_TYPE", "Send uncompressed CSV bytes as Content-Type: text/csv.", 415)
        declared = request.headers.get("content-length")
        if declared is not None:
            if not declared.isascii() or not declared.isdigit():
                raise ApplicationError("INVALID_REQUEST", "Content-Length must be a nonnegative byte count.", 400)
            # Compare length first to avoid arbitrary-length integer conversion.
            if len(declared) > 16 or int(declared) > MAX_BYTES:
                raise ApplicationError("FILE_TOO_LARGE", "Maximum CSV upload size is 10 MiB.", 413)
        spool_dir = Path(tempfile.gettempdir()).resolve()
        if spool_dir.is_relative_to(service.root.resolve()):
            raise ApplicationError("INTERNAL_ERROR", "Server temporary storage must be outside the repository.", 503)
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b", dir=spool_dir) as source:
            size = 0
            digest = hashlib.sha256()
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ApplicationError("FILE_TOO_LARGE", "Maximum CSV upload size is 10 MiB.", 413)
                source.write(chunk)
                digest.update(chunk)
            if declared is not None and size != int(declared):
                raise ApplicationError("INVALID_REQUEST", "Body size does not match Content-Length.", 400)
            if size == 0:
                raise ApplicationError("EMPTY_FILE", "The uploaded CSV is empty.", 422)
            # AnyIO's default non-abandoning worker keeps the spool and lock alive
            # until processing finishes, including on request cancellation.
            return await run_in_threadpool(service.process, source, filename=filename, size=size,
                                           sha256=digest.hexdigest(), analyze=analyze)
    except ClientDisconnect as exc:
        raise ApplicationError("INVALID_REQUEST", "CSV upload was interrupted.", 400) from exc
    finally:
        if acquired:
            service.lock.release()


@router.post("/validate", response_model=ValidationResponse, openapi_extra=UPLOAD_DOC)
async def validate(request: Request):
    return await uploaded(request, analyze=False)


@router.post("/analyze", response_model=AnalysisResponse, openapi_extra=UPLOAD_DOC)
async def analyze(request: Request):
    return await uploaded(request, analyze=True)
