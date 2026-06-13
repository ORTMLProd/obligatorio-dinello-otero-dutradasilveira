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
from src.api.routers import health, metrics, model_info, predict
from src.config import get_settings
from src.models.export import load_bundle
from src.monitoring.logging import configure_inference_logging
from src.monitoring.metrics import record_request, set_training_distribution

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Structured inference logs to stdout (Fase 3.4) — configured once at startup.
    configure_inference_logging()
    # Load the model bundle (estimator + fitted preprocessor) once at startup and keep it
    # on app.state. The preprocessor is never re-fitted here (invariant 3). If no bundle is
    # present (fresh deploy before training), the API still boots and /predict returns 503.
    model_dir = get_settings().resolved_model_dir()
    try:
        app.state.bundle = load_bundle(model_dir)
        logger.info("loaded model bundle from %s (%s)", model_dir, app.state.bundle.model_version)
        # Publish the train-split class distribution as the drift baseline (Fase 3.4).
        set_training_distribution(app.state.bundle.train_class_ratio)
    except (FileNotFoundError, OSError):
        app.state.bundle = None
        logger.warning("no model bundle at %s — /predict will return 503", model_dir)
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
    app.include_router(predict.router)
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
