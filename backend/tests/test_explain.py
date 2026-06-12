"""Tests for SHAP explanations (Phase 3 explainability electivo).

Explanations are computed via XGBoost's native TreeSHAP (``pred_contribs``), so these
tests need no ``shap`` library. The opaque 512-dim ResNet embedding is collapsed into a
single ``visual_embedding`` bucket; one-hot ``league`` columns fold back to ``league``.
The grouping must conserve total contribution (no mass lost) — a TreeSHAP additivity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import build_preprocessor
from src.features.tabular import TABULAR_COLUMNS
from src.models.explain import (
    VISUAL_FEATURE,
    build_feature_groups,
    explain_prediction,
    grouped_contributions,
    shap_values_per_class,
)
from src.models.export import ModelBundle, predict_frame
from src.models.train import ModelSpec, SplitData, fit_one

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
            embs.append(rng.normal(loc=idx, scale=0.3, size=EMB_DIM))
    tabular = pd.DataFrame(rows, columns=list(TABULAR_COLUMNS))
    return SplitData(tabular=tabular, embedding=np.array(embs, dtype=np.float32), y=np.array(ys))


def _xgb_bundle() -> tuple[ModelBundle, SplitData]:
    train = _synthetic_split(20, seed=1)
    splits = {"train": train, "val": train, "test": train}
    pre = build_preprocessor(scale_numeric=False).fit(train.tabular)
    spec = ModelSpec(type="xgboost", params={"max_depth": 3, "n_estimators": 30})
    estimator = fit_one(spec, splits, pre, use_embedding=True, seed=42)
    bundle = ModelBundle(
        model=estimator,
        preprocessor=pre,
        classes=CLASSES,
        tabular_columns=list(TABULAR_COLUMNS),
        embedding_dim=EMB_DIM,
        model_type="xgboost",
        model_version="test",
        dataset_hash="d",
        train_config_hash="c",
        metrics={},
    )
    return bundle, train


def test_build_feature_groups_collapses_embedding_and_onehot() -> None:
    bundle, _ = _xgb_bundle()
    order, groups = build_feature_groups(bundle.preprocessor, bundle.embedding_dim)

    assert order[-1] == VISUAL_FEATURE  # visual bucket is last
    assert order.count("league") == 1  # one-hot league folded to a single group
    assert len(groups[VISUAL_FEATURE]) == EMB_DIM  # all embedding dims in one bucket

    # The groups partition the feature matrix columns exactly (0..N-1, no gaps/overlaps).
    all_idx = sorted(i for idxs in groups.values() for i in idxs)
    assert all_idx == list(range(len(all_idx)))


def test_grouping_conserves_total_contribution() -> None:
    """Folding columns into groups must not change the total SHAP mass (additivity)."""
    bundle, train = _xgb_bundle()
    row_tab, row_emb = train.tabular.iloc[[0]], train.embedding[[0]]

    grouped = grouped_contributions(bundle, row_tab, row_emb)[0]
    per_class = shap_values_per_class(bundle, row_tab, row_emb)  # (1, n_features, n_classes)
    _, proba = predict_frame(bundle, row_tab, row_emb)
    cls = int(proba.argmax())
    raw_sum = float(per_class[0, :, cls].sum())

    assert abs(sum(grouped.values()) - raw_sum) < 1e-4


def test_explain_prediction_returns_named_contributions() -> None:
    bundle, train = _xgb_bundle()
    expl = explain_prediction(bundle, train.tabular.iloc[[0]], train.embedding[[0]])

    assert VISUAL_FEATURE in expl
    assert any(k in TABULAR_COLUMNS for k in expl)  # at least one named tabular feature
    assert all(isinstance(v, float) for v in expl.values())


def test_explain_prediction_top_k_sorted_by_magnitude() -> None:
    bundle, train = _xgb_bundle()
    expl = explain_prediction(bundle, train.tabular.iloc[[0]], train.embedding[[0]], top_k=3)

    assert len(expl) == 3
    vals = [abs(v) for v in expl.values()]
    assert vals == sorted(vals, reverse=True)
