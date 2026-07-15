"""Pydantic schemas — the strict contract of the API (invariant 4).

Every response/request model forbids extra fields (``extra="forbid"``). The clip model is
visual-only: ``/predict/clip`` and ``/predict/clip/batch`` receive video upload(s) and the
extractor (frames → CNN) is identical in training and serving (no training-serving skew).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness payload returned by ``GET /health``."""

    model_config = ConfigDict(extra="forbid")

    status: str
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded model."""

    # ``protected_namespaces=()`` silences pydantic's warning about the ``model_`` prefix.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_loaded: bool
    version: str | None
    message: str
    model_type: str | None = None
    classes: list[str] | None = None
    test_macro_f1: float | None = None


class GradcamFrame(BaseModel):
    """One Grad-CAM overlay: the frame index and a base64-encoded JPG of the overlay."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int
    image_base64: str


class ClipPredictResponse(BaseModel):
    """Prediction for an uploaded video clip: class, probabilities and Grad-CAM overlays.

    Powers ``POST /predict/clip`` (online): the visual-only clip model classifies the clip
    and returns a Grad-CAM overlay (base64 JPG) per sampled frame for the predicted class.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    gradcam: list[GradcamFrame]


class ClipBatchItem(BaseModel):
    """One clip's result within a batch (no Grad-CAM — too heavy per clip in bulk)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    filename: str | None
    predicted_label: str
    probabilities: dict[str, float]
    model_version: str


class ClipBatchPredictResponse(BaseModel):
    """Per-clip results, aligned with the uploaded files order (``POST /predict/clip/batch``)."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[ClipBatchItem]
