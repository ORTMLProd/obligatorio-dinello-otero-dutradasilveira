"""Pydantic schemas — the strict contract of the API (invariant 4).

Every response/request model forbids extra fields (``extra="forbid"``). The same
``PredictRequest``/``PredictResponse`` types back both ``/predict`` and ``/predict/batch``.

Visual contract (v0): the request carries the *pre-extracted* ResNet embedding
(``resnet_features``), not a raw image. The model is trained on SoccerNet's pooled
ResNet+PCA features, which we cannot reproduce from pixels at serving time; consuming the
same embedding avoids training-serving skew (invariant 3). Raw-image ingestion arrives in
v1 with our own CNN, where the extractor is identical in training and serving.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Pooled ResNet+PCA dimensionality of the v0 dataset — used only for a submittable
# Swagger example, not for validation (the router checks against the loaded bundle).
_EMBEDDING_EXAMPLE_DIM = 512


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


class PredictRequest(BaseModel):
    """One window to classify: point-in-time tabular features + pooled ResNet embedding.

    The tabular fields mirror ``src.features.tabular.TABULAR_COLUMNS`` exactly. The caller
    supplies the current-state values (the same quantities the dataset builder computed
    point-in-time for training).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "half": 2,
                "minute": 44,
                "score_diff": 1,
                "league": "england_epl",
                "team_is_home": 1,
                "visible": 1,
                "events_so_far": 27,
                "secs_since_last_event": 18.0,
                "resnet_features": [0.0] * _EMBEDDING_EXAMPLE_DIM,
            }
        },
    )

    half: int = Field(description="Half the window belongs to (1 or 2).", ge=1, le=2)
    minute: int = Field(description="Minute within the half (position_ms // 60000).", ge=0)
    score_diff: float = Field(description="Accumulated home − away goals, point-in-time.")
    league: str = Field(description="Competition id, e.g. 'england_epl'.")
    team_is_home: int = Field(description="1 home, 0 away, -1 not applicable.", ge=-1, le=1)
    visible: int = Field(description="Whether the action is visible on screen (0/1).", ge=0, le=1)
    events_so_far: int = Field(description="Annotated events strictly before t.", ge=0)
    secs_since_last_event: float = Field(
        description="Seconds since the previous event; -1 if none."
    )
    resnet_features: list[float] = Field(
        description="Pre-extracted pooled ResNet embedding; length must match the model.",
        min_length=1,
    )


class PredictResponse(BaseModel):
    """Predicted class, full per-class probabilities, model version and (optionally) SHAP.

    ``explanations`` decomposes the prediction into per-feature contributions (TreeSHAP) for
    the predicted class: each tabular feature by name, plus a single ``visual_embedding``
    bucket aggregating the 512 ResNet dimensions (opaque individually). Positive values push
    toward the predicted class. ``None`` when explanations are disabled or unavailable for the
    model type.
    """

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_label": "corner",
                "probabilities": {
                    "background": 0.07,
                    "card": 0.03,
                    "corner": 0.71,
                    "goal": 0.05,
                    "substitution": 0.14,
                },
                "model_version": "v1-tuned-xgboost-91c40640",
                "explanations": {"visual_embedding": 1.42, "league": 0.31, "minute": -0.12},
            }
        },
    )

    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    explanations: dict[str, float] | None = None


class BatchPredictRequest(BaseModel):
    """A batch of windows to classify in a single call (sync in v0)."""

    model_config = ConfigDict(extra="forbid")

    items: list[PredictRequest] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    """Per-item results, aligned with the request order."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[PredictResponse]


class GradcamFrame(BaseModel):
    """One Grad-CAM overlay: the frame index and a base64-encoded JPG of the overlay."""

    model_config = ConfigDict(extra="forbid")

    frame_index: int
    image_base64: str


class ClipPredictResponse(BaseModel):
    """Prediction for an uploaded video clip: class, probabilities and Grad-CAM overlays.

    Powers ``POST /predict/clip`` (Fase 3.5): the visual-only clip model classifies the clip
    and returns a Grad-CAM overlay (base64 JPG) per sampled frame for the predicted class.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    gradcam: list[GradcamFrame]
