"""FastAPI application entrypoint.

Wires the routers and, from Fase 2 on, will load the model artifact and the fitted
feature transformers inside the ``lifespan``. All preprocessing lives in
``src.features`` so training and serving share a single source of truth, preventing
training-serving skew.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src import __version__
from src.api.routers import clip_predict, health, metrics, model_info
from src.config import get_settings
from src.models.clip_export import load_clip_bundle
from src.models.clip_model import pick_device
from src.monitoring.logging import configure_inference_logging
from src.monitoring.metrics import record_request, set_training_distribution

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Structured inference logs to stdout (Fase 3.4) — configured once at startup.
    configure_inference_logging()
    # Load the clip model bundle once at startup and keep it on app.state. The eval transform
    # is reconstructed from the serialized stats (invariant 3, no re-fit). If no bundle is
    # present (fresh deploy before training), the API still boots and /predict/clip returns 503.
    clip_dir = get_settings().resolved_clip_model_dir()
    try:
        device = pick_device()
        app.state.clip_model, app.state.clip_meta = load_clip_bundle(clip_dir, device)
        app.state.clip_device = device
        logger.info("loaded clip model from %s (%s)", clip_dir, app.state.clip_meta.model_version)
        # Publish the train-split class distribution as the drift baseline (Fase 3.4).
        if app.state.clip_meta.train_class_ratio:
            set_training_distribution(app.state.clip_meta.train_class_ratio)
    except (FileNotFoundError, OSError):
        app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
        logger.warning("no clip model at %s — /predict/clip will return 503", clip_dir)
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
    app.include_router(clip_predict.router)
    app.include_router(metrics.router)

    @app.middleware("http")
    async def _count_requests(request: Request, call_next):
        # Traffic + errors per endpoint. Paths are fixed (no path params), so label
        # cardinality stays bounded. Uses the route path when matched, else the raw path.
        response = await call_next(request)
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        record_request(endpoint, response.status_code)
        return response

    @app.get("/", tags=["root"])
    async def root() -> dict[str, str]:
        return {"service": settings.service_name, "version": __version__, "status": "ok"}

    return app


app = create_app()
