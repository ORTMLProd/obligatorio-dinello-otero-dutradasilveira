"""Smoke test for the training core: fit → evaluate → export → reload → predict.

Exercises the deterministic path without MLflow or the real dataset (synthetic windows),
so it stays fast and CI-friendly. The full ``run()`` orchestration with MLflow is verified
live. Guards the contract the API depends on: a reloaded bundle predicts valid classes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import build_preprocessor
from src.features.tabular import TABULAR_COLUMNS
from src.models.export import ModelBundle, load_bundle, predict_frame, save_bundle
from src.models.train import ModelSpec, SplitData, evaluate_on, fit_one

CLASSES = ["background", "card", "corner", "goal", "substitution"]
EMB_DIM = 8


def _synthetic_split(n_per_class: int, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    rows, ys, embs = [], [], []
    for idx, _cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            rows.append(
                {
                    "half": int(rng.integers(1, 3)),
                    "minute": int(rng.integers(0, 50)),
                    "score_diff": int(rng.integers(-2, 3)),
                    "league": "england_epl",
                    "team_is_home": int(rng.integers(-1, 2)),
                    "visible": int(rng.integers(0, 2)),
                    "events_so_far": int(rng.integers(0, 20)),
                    "secs_since_last_event": float(rng.integers(-1, 300)),
                }
            )
            ys.append(idx)
            # Give each class a separable embedding cluster so the model learns something.
            embs.append(rng.normal(loc=idx, scale=0.3, size=EMB_DIM))
    tabular = pd.DataFrame(rows, columns=list(TABULAR_COLUMNS))
    return SplitData(tabular=tabular, embedding=np.array(embs, dtype=np.float32), y=np.array(ys))


def test_train_core_exports_reloadable_bundle(tmp_path) -> None:
    train = _synthetic_split(n_per_class=20, seed=1)
    test = _synthetic_split(n_per_class=5, seed=2)
    splits = {"train": train, "val": test, "test": test}

    preprocessor = build_preprocessor(scale_numeric=True).fit(train.tabular)
    spec = ModelSpec(
        type="logistic_regression", params={"max_iter": 500, "class_weight": "balanced"}
    )
    estimator = fit_one(spec, splits, preprocessor, use_embedding=True, seed=42)

    metrics = evaluate_on(estimator, test, preprocessor, use_embedding=True, classes=CLASSES)
    # Every target class must appear in the per-class report (invariant 5).
    assert set(metrics["per_class"]) == set(CLASSES)
    assert 0.0 <= metrics["macro_f1"] <= 1.0

    bundle = ModelBundle(
        model=estimator,
        preprocessor=preprocessor,
        classes=CLASSES,
        tabular_columns=list(TABULAR_COLUMNS),
        embedding_dim=EMB_DIM,
        model_type=spec.type,
        model_version="v0-test",
        dataset_hash="deadbeef",
        train_config_hash="cafe",
        metrics=metrics,
    )
    save_bundle(bundle, tmp_path)
    reloaded = load_bundle(tmp_path)

    labels, proba = predict_frame(reloaded, test.tabular, test.embedding)
    assert len(labels) == len(test.tabular)
    assert all(lbl in CLASSES for lbl in labels)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
