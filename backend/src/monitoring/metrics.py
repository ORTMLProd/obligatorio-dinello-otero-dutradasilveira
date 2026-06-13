"""Prometheus metrics for the serving API (Fase 3.4 — observabilidad).

Metrics are defined explicitly (not auto-instrumented) so it is clear what is measured and
why — the course values being able to defend each decision. Exposed at ``GET /metrics``:

- ``soccernet_predictions_total{predicted_label, model_version}`` — predictions per class;
  the live distribution that, compared against the train baseline, signals drift.
- ``soccernet_prediction_latency_seconds`` — inference latency histogram (ms-scale buckets).
- ``soccernet_requests_total{endpoint, status}`` — traffic and errors per endpoint.
- ``soccernet_training_class_ratio{class_name}`` — train-split class proportions (drift
  baseline), set once at startup from the loaded bundle.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

CONTENT_TYPE = CONTENT_TYPE_LATEST

# Latency buckets tuned to the ms-scale of this model (default buckets start at 5ms, too coarse).
_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)

PREDICTIONS = Counter(
    "soccernet_predictions",
    "Predicciones de inferencia por clase y versión de modelo.",
    ["predicted_label", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "soccernet_prediction_latency_seconds",
    "Latencia de inferencia por request (segundos).",
    buckets=_LATENCY_BUCKETS,
)
REQUESTS = Counter(
    "soccernet_requests",
    "Requests HTTP por endpoint y código de estado.",
    ["endpoint", "status"],
)
TRAINING_CLASS_RATIO = Gauge(
    "soccernet_training_class_ratio",
    "Proporción de cada clase en el split de entrenamiento (baseline de drift).",
    ["class_name"],
)


def record_prediction(predicted_label: str, model_version: str) -> None:
    """Bump the per-class prediction counter (one call per predicted item)."""
    PREDICTIONS.labels(predicted_label=predicted_label, model_version=model_version).inc()


def observe_latency(latency_s: float) -> None:
    """Observe one request's inference latency (one call per request, not per item)."""
    PREDICTION_LATENCY.observe(latency_s)


def record_request(endpoint: str, status: int | str) -> None:
    """Count one HTTP request by endpoint and status code."""
    REQUESTS.labels(endpoint=endpoint, status=str(status)).inc()


def set_training_distribution(ratios: dict[str, float]) -> None:
    """Publish the train-split class distribution as the drift baseline gauge."""
    for class_name, ratio in ratios.items():
        TRAINING_CLASS_RATIO.labels(class_name=class_name).set(ratio)


def render() -> tuple[bytes, str]:
    """Serialize all metrics in Prometheus text format. Returns ``(body, content_type)``."""
    return generate_latest(), CONTENT_TYPE
