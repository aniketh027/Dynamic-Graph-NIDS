"""One sanitized error envelope for every application error."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from backend.app.schemas.contracts import APIError, ErrorCode, ErrorResponse

logger = logging.getLogger("graphnids.errors")


class ApplicationError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 503):
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


def envelope(code: ErrorCode, message: str, status: int) -> JSONResponse:
    payload = ErrorResponse(error=APIError(code=code, message=message))
    return JSONResponse(payload.model_dump(mode="json"), status_code=status)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error(_: Request, exc: ApplicationError):
        return envelope(exc.code, exc.message, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError):
        logger.warning("Rejected request validation: %s", type(exc).__name__)
        return envelope("INVALID_REQUEST", "The request does not match the API contract.", 422)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        code: ErrorCode = "CAPABILITY_NOT_IMPLEMENTED" if exc.status_code in {404, 405} else "INVALID_REQUEST"
        return envelope(code, "This API operation is unavailable.", exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        logger.error("Internal request failure", exc_info=exc)
        return envelope("INTERNAL_ERROR", "An internal application error occurred.", 500)
