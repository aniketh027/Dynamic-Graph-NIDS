"""Run from the repository root: python -m uvicorn backend.app.main:app."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.api.replay import router as replay_router
from backend.app.core.config import Settings
from backend.app.core.errors import install_error_handlers
from backend.app.services.inference import FrozenInferenceService
from backend.app.services.replay import ReplayService


def create_app(settings: Settings | None = None, service: FrozenInferenceService | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    service = service or FrozenInferenceService(settings.repository_root)
    replay = ReplayService(settings.repository_root, enabled=settings.load_model)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service.initialize(load_model=settings.load_model)
        yield
        service.close()
        replay.close()

    application = FastAPI(title="GraphNIDS local application API", version="1.0.0", lifespan=lifespan)
    application.state.inference = service
    application.state.replay = replay
    install_error_handlers(application)
    application.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                               allow_credentials=False, allow_methods=["GET", "POST"],
                               allow_headers=["Accept", "Content-Type", "X-Filename"])
    application.include_router(router)
    application.include_router(replay_router)
    return application


app = create_app()
