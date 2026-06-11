"""FastAPI application entrypoint.

Wires the routers and, from Fase 2 on, will load the model artifact and the fitted
feature transformers inside the ``lifespan``. All preprocessing lives in
``src.features`` so training and serving share a single source of truth, preventing
training-serving skew.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import __version__
from src.api.routers import health, model_info
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Fase 2: load model + fitted transformers here and store them on ``app.state``.
    yield


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title=settings.service_name,
        version=__version__,
        summary="Clasificador de eventos de SoccerNet — API de inferencia.",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(model_info.router)

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {"service": settings.service_name, "version": __version__, "status": "ok"}

    return app


app = create_app()
