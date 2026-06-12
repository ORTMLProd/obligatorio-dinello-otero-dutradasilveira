"""Inference endpoints: online (``/predict``) and batch (``/predict/batch``).

Both reuse ``src.models.export.predict_frame`` and ``src.features.preprocess`` — the same
code path as training — so serving and training assemble features and read probabilities
identically (invariant 3, no training-serving skew). The fitted preprocessor and model
come from the bundle loaded into ``app.state`` at startup; nothing is re-fitted here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from src.models.explain import grouped_contributions
from src.models.export import ModelBundle, predict_frame

router = APIRouter(tags=["inference"])

# Models whose predictions we can decompose with native TreeSHAP (see src.models.explain).
_EXPLAINABLE_MODEL_TYPES = {"xgboost"}


def _require_bundle(request: Request) -> ModelBundle:
    """Return the loaded bundle or 503 if no model is served (fresh deploy before train)."""
    bundle = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded — train and mount models/v0.")
    return bundle


def _to_frame_and_embedding(
    items: list[PredictRequest], bundle: ModelBundle
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Build the tabular frame and embedding matrix the model expects from the requests."""
    frame = pd.DataFrame(
        [{col: getattr(it, col) for col in bundle.tabular_columns} for it in items],
        columns=bundle.tabular_columns,
    )
    if bundle.embedding_dim is None:
        return frame, None
    for i, it in enumerate(items):
        if len(it.resnet_features) != bundle.embedding_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"item {i}: resnet_features has length {len(it.resnet_features)}, "
                    f"expected {bundle.embedding_dim}."
                ),
            )
    embedding = np.array([it.resnet_features for it in items], dtype=np.float32)
    return frame, embedding


def _to_responses(
    items: list[PredictRequest], bundle: ModelBundle, explain: bool
) -> list[PredictResponse]:
    frame, embedding = _to_frame_and_embedding(items, bundle)
    labels, proba = predict_frame(bundle, frame, embedding)

    # SHAP for the whole batch in a single pass (one TreeSHAP call covers all rows), only
    # when requested and supported. Same preprocessing/inference path as predict_frame.
    explanations: list[dict[str, float] | None] = [None] * len(items)
    if explain and bundle.model_type in _EXPLAINABLE_MODEL_TYPES:
        explanations = list(grouped_contributions(bundle, frame, embedding))

    return [
        PredictResponse(
            predicted_label=labels[i],
            probabilities=dict(zip(bundle.classes, proba[i].tolist(), strict=True)),
            model_version=bundle.model_version,
            explanations=explanations[i],
        )
        for i in range(len(items))
    ]


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request, explain: bool = True) -> PredictResponse:
    """Classify a single window into one of the event classes (or background).

    Set ``?explain=false`` to skip the SHAP decomposition (lower latency).
    """
    bundle = _require_bundle(request)
    return _to_responses([req], bundle, explain=explain)[0]


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    req: BatchPredictRequest, request: Request, explain: bool = False
) -> BatchPredictResponse:
    """Classify a batch of windows in one call (synchronous in v0).

    SHAP is off by default for batch throughput; set ``?explain=true`` to include it.
    """
    bundle = _require_bundle(request)
    return BatchPredictResponse(predictions=_to_responses(req.items, bundle, explain=explain))
