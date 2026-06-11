"""Model metadata endpoint (stub in Fase 0; the real model arrives in Fase 2)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return info about the loaded model. No model is served in Fase 0."""
    return ModelInfoResponse(
        model_loaded=False,
        version=None,
        message="No model loaded in Fase 0 — the baseline (v0) arrives in Fase 2.",
    )
