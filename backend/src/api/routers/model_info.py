"""Model metadata endpoint — reports the bundle loaded into app.state at startup."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently served model, or that none is loaded."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        return ModelInfoResponse(
            model_loaded=False,
            version=None,
            message="No model loaded — train (python -m src.models.train) and mount models/v0.",
        )
    return ModelInfoResponse(
        model_loaded=True,
        version=bundle.model_version,
        message="Model loaded and serving predictions.",
        model_type=bundle.model_type,
        classes=bundle.classes,
        test_macro_f1=bundle.metrics.get("macro_f1"),
    )
