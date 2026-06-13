"""Structured inference logging (Fase 3.4).

One JSON line per inference to stdout (12-factor: the platform collects logs; the container
keeps no state). Records the tabular features, predicted class, probabilities, model version
and latency — **never** the ResNet embedding or any image (data policy / NDA). This is the
audit trail behind the drift signal exposed via Prometheus.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

logger = logging.getLogger("soccernet.inference")


def configure_inference_logging() -> None:
    """Send the inference logger to stdout at INFO (called once at startup).

    A bare custom logger has no handler, so its records would be dropped (uvicorn only
    configures its own loggers). We attach a stdout handler emitting the raw JSON line and
    stop propagation so the record is not reformatted by other handlers. Idempotent.
    """
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def build_inference_record(
    features: dict[str, Any],
    predicted_label: str,
    probabilities: dict[str, float],
    model_version: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Build the structured inference record. ``features`` must already exclude the embedding."""
    return {
        "event": "inference",
        "model_version": model_version,
        "predicted_label": predicted_label,
        "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
        "latency_ms": latency_ms,
        "features": features,
    }


def log_inference(
    features: dict[str, Any],
    predicted_label: str,
    probabilities: dict[str, float],
    model_version: str,
    latency_ms: float,
) -> None:
    """Emit one structured inference line to stdout (best-effort; never raises to the caller)."""
    try:
        record = build_inference_record(
            features, predicted_label, probabilities, model_version, latency_ms
        )
        logger.info(json.dumps(record, ensure_ascii=False))
    except Exception:  # logging must never break serving
        logger.debug("failed to log inference record", exc_info=True)
