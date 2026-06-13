"""``GET /metrics`` — Prometheus exposition endpoint (Fase 3.4)."""

from __future__ import annotations

from fastapi import APIRouter, Response

from src.monitoring.metrics import render

router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def metrics() -> Response:
    """Expose all metrics in Prometheus text format (scraped by the prometheus service)."""
    body, content_type = render()
    return Response(content=body, media_type=content_type)
