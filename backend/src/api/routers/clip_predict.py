"""Clip inference endpoint: upload a video -> class + probabilities + Grad-CAM (Fase 3.5)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from src.api.schemas import ClipPredictResponse, GradcamFrame
from src.monitoring.metrics import record_prediction
from src.serving.clip_inference import serve_clip

router = APIRouter(tags=["inference"])


def _require_clip(request: Request):
    model = getattr(request.app.state, "clip_model", None)
    meta = getattr(request.app.state, "clip_meta", None)
    if model is None or meta is None:
        raise HTTPException(
            status_code=503,
            detail="No clip model loaded — train and mount models/clips-v1.",
        )
    return model, meta, getattr(request.app.state, "clip_device", None)


@router.post("/predict/clip", response_model=ClipPredictResponse)
async def predict_clip_endpoint(request: Request, video: UploadFile) -> ClipPredictResponse:
    """Classify an uploaded video clip and return per-frame Grad-CAM overlays (base64 JPG)."""
    model, meta, device = _require_clip(request)
    data = await video.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty video upload")
    suffix = Path(video.filename or "").suffix or ".mp4"
    try:
        label, proba, overlays = serve_clip(model, meta, data, device, suffix=suffix)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not read video: {exc}") from exc
    record_prediction(label, meta.model_version)
    return ClipPredictResponse(
        predicted_label=label,
        probabilities=dict(zip(meta.classes, proba.tolist(), strict=True)),
        model_version=meta.model_version,
        gradcam=[GradcamFrame(frame_index=i, image_base64=b) for i, b in enumerate(overlays)],
    )
