"""Tests for monitoring (Fase 3.4): Prometheus metrics + structured inference logging.

Metrics are read back from the default registry via ``get_sample_value`` (counters/gauges
accumulate, so we assert on the delta). The inference record must never carry the embedding
or any image (data policy / NDA).
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from src.monitoring.logging import build_inference_record
from src.monitoring.metrics import (
    observe_latency,
    record_prediction,
    render,
    set_training_distribution,
)


def _pred_count(label: str, version: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "soccernet_predictions_total",
            {"predicted_label": label, "model_version": version},
        )
        or 0.0
    )


def test_record_prediction_increments_class_counter() -> None:
    before = _pred_count("goal", "vtest")
    record_prediction("goal", "vtest")
    assert _pred_count("goal", "vtest") == before + 1


def test_observe_latency_increments_histogram_count() -> None:
    name = "soccernet_prediction_latency_seconds_count"
    before = REGISTRY.get_sample_value(name) or 0.0
    observe_latency(0.003)
    assert (REGISTRY.get_sample_value(name) or 0.0) == before + 1


def test_set_training_distribution_sets_gauge() -> None:
    set_training_distribution({"goal": 0.1, "background": 0.6})
    assert (
        REGISTRY.get_sample_value("soccernet_training_class_ratio", {"class_name": "goal"}) == 0.1
    )
    assert (
        REGISTRY.get_sample_value("soccernet_training_class_ratio", {"class_name": "background"})
        == 0.6
    )


def test_render_exposes_metric_names_and_prometheus_content_type() -> None:
    body, content_type = render()
    text = body.decode()
    assert "soccernet_predictions_total" in text
    assert "soccernet_training_class_ratio" in text
    assert "text/plain" in content_type


def test_inference_record_has_fields_and_no_embedding() -> None:
    rec = build_inference_record(
        features={"half": 2, "minute": 40, "league": "england_epl"},
        predicted_label="corner",
        probabilities={"corner": 0.7, "goal": 0.3},
        model_version="v1-tuned",
        latency_ms=2.345,
    )
    assert rec["event"] == "inference"
    assert rec["predicted_label"] == "corner"
    assert rec["model_version"] == "v1-tuned"
    assert rec["features"]["league"] == "england_epl"
    assert rec["latency_ms"] == 2.345
    # The embedding / any image must never be logged (data policy).
    assert "resnet_features" not in str(rec)
