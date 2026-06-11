"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from src import __version__
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service liveness and version.

    Consumed by the frontend (to show backend status) and by the Docker healthcheck.
    """
    return HealthResponse(status="ok", version=__version__)
