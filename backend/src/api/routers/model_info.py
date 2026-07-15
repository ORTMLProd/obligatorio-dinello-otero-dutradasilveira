"""Model metadata endpoint — reports the clip model loaded into app.state at startup."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.schemas import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    """Return metadata about the currently served clip model, or that none is loaded."""
    meta = getattr(request.app.state, "clip_meta", None)
    if meta is None:
        return ModelInfoResponse(
            model_loaded=False,
            version=None,
            message="No clip model loaded — train (python -m src.models.train_clips) "
            "and mount models/clips-v1.",
        )
    return ModelInfoResponse(
        model_loaded=True,
        version=meta.model_version,
        message="Clip model loaded and serving predictions.",
        model_type=f"clip-cnn-{meta.backbone}",
        classes=meta.classes,
        test_macro_f1=meta.metrics.get("macro_f1"),
    )
