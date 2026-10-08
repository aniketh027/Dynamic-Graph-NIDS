from fastapi import APIRouter, Request

from backend.app.schemas.contracts import (
    CapabilitiesResponse, ErrorResponse, HealthResponse, ModelResponse, SchemaResponse, StatusResponse,
)
from backend.app.services.inference import schema_metadata

router = APIRouter(prefix="/api/v1", responses={503: {"model": ErrorResponse}})


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    return request.app.state.inference.status().model_copy(update={"inference_ready": request.app.state.replay.ready})


@router.get("/model", response_model=ModelResponse)
def model(request: Request) -> ModelResponse:
    return request.app.state.inference.metadata()


@router.get("/schema", response_model=SchemaResponse)
def schema() -> SchemaResponse:
    return schema_metadata()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request) -> CapabilitiesResponse:
    return CapabilitiesResponse(model_inference_adapter=request.app.state.inference.state)
