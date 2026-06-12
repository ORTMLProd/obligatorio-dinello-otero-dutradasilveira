"""Contract tests for /predict and /predict/batch (invariant 4 — strict API contract).

A tiny model bundle is injected into ``app.state`` so the endpoints can be exercised
without the real trained artifact (CI-friendly, fresh-clone safe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.features.preprocess import build_preprocessor
from src.features.tabular import TABULAR_COLUMNS
from src.models.export import ModelBundle
from src.models.train import ModelSpec, SplitData, fit_one

CLASSES = ["background", "card", "corner", "goal", "substitution"]
EMB_DIM = 4


def _tiny_bundle() -> ModelBundle:
    rng = np.random.default_rng(0)
    rows, ys, embs = [], [], []
    for idx in range(len(CLASSES)):
        for _ in range(15):
            rows.append(
                {
                    "half": 1,
                    "minute": int(rng.integers(0, 45)),
                    "score_diff": int(rng.integers(-2, 3)),
                    "league": "england_epl",
                    "team_is_home": int(rng.integers(-1, 2)),
                    "visible": 1,
                    "events_so_far": int(rng.integers(0, 10)),
                    "secs_since_last_event": float(rng.integers(-1, 200)),
                }
            )
            ys.append(idx)
            embs.append(rng.normal(loc=idx, scale=0.3, size=EMB_DIM))
    tabular = pd.DataFrame(rows, columns=list(TABULAR_COLUMNS))
    split = SplitData(tabular=tabular, embedding=np.array(embs, dtype=np.float32), y=np.array(ys))
    pre = build_preprocessor(scale_numeric=True).fit(tabular)
    spec = ModelSpec(type="logistic_regression", params={"max_iter": 500})
    estimator = fit_one(spec, {"train": split}, pre, use_embedding=True, seed=0)
    return ModelBundle(
        model=estimator,
        preprocessor=pre,
        classes=CLASSES,
        tabular_columns=list(TABULAR_COLUMNS),
        embedding_dim=EMB_DIM,
        model_type="logistic_regression",
        model_version="v0-test",
        dataset_hash="x",
        train_config_hash="y",
        metrics={"macro_f1": 0.5},
    )


def _payload(**overrides) -> dict:
    base = {
        "half": 2,
        "minute": 44,
        "score_diff": 1,
        "league": "england_epl",
        "team_is_home": 1,
        "visible": 1,
        "events_so_far": 27,
        "secs_since_last_event": 18.0,
        "resnet_features": [0.1, 0.2, 0.3, 0.4],
    }
    base.update(overrides)
    return base


@pytest.fixture
def client_with_model() -> TestClient:
    app.state.bundle = _tiny_bundle()
    yield TestClient(app)
    app.state.bundle = None


def test_predict_returns_label_and_probabilities(client_with_model: TestClient) -> None:
    resp = client_with_model.post("/predict", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"predicted_label", "probabilities", "model_version"}
    assert body["predicted_label"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert body["probabilities"][body["predicted_label"]] == pytest.approx(
        max(body["probabilities"].values())
    )
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert body["model_version"] == "v0-test"


def test_predict_rejects_extra_field(client_with_model: TestClient) -> None:
    resp = client_with_model.post("/predict", json=_payload(unexpected="boom"))
    assert resp.status_code == 422  # extra="forbid"


def test_predict_rejects_wrong_embedding_length(client_with_model: TestClient) -> None:
    resp = client_with_model.post("/predict", json=_payload(resnet_features=[0.1, 0.2]))
    assert resp.status_code == 422


def test_predict_batch_returns_aligned_results(client_with_model: TestClient) -> None:
    resp = client_with_model.post(
        "/predict/batch", json={"items": [_payload(), _payload(league="spain_laliga")]}
    )
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2
    assert all(p["predicted_label"] in CLASSES for p in preds)


def test_predict_503_when_no_model() -> None:
    app.state.bundle = None
    client = TestClient(app)
    resp = client.post("/predict", json=_payload())
    assert resp.status_code == 503
