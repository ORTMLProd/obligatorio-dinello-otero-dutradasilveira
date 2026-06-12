"""SHAP explanations for the classifier — Phase 3 explainability electivo.

Why TreeSHAP via XGBoost, not the ``shap`` library: the served model is gradient-boosted
trees, and XGBoost computes exact TreeSHAP natively (``pred_contribs=True``). Using it keeps
the serving image free of ``shap``'s heavy ``numba``/``llvmlite`` stack — the API explains its
own predictions cheaply. The ``shap`` library is used only in the notebook, for plots.

The model input is ``[tabular ⊕ embedding]`` (invariant 3). The 512-dim ResNet embedding is
not human-interpretable dimension by dimension, so its contributions are collapsed into one
``visual_embedding`` bucket; one-hot ``league`` columns fold back to ``league``. The result is
a readable decomposition: how much the *visual* signal drove the prediction vs each *tabular*
context feature. This module is the single source of explanation logic, imported by both the
API and the notebook (never duplicated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import CATEGORICAL_COLUMNS, assemble_matrix
from src.models.export import ModelBundle, predict_frame

VISUAL_FEATURE = "visual_embedding"


def _to_original_name(transformed_name: str) -> str:
    """Map a preprocessor output column back to its original tabular feature.

    ``get_feature_names_out`` yields ``cat__league_england_epl``, ``num__minute``,
    ``pass__half``. Strip the transformer prefix, then fold a one-hot like ``league_<value>``
    back to its source categorical (``league``).
    """
    name = transformed_name.split("__", 1)[1] if "__" in transformed_name else transformed_name
    for cat in CATEGORICAL_COLUMNS:
        if name.startswith(f"{cat}_"):
            return cat
    return name


def build_feature_groups(
    preprocessor, embedding_dim: int | None
) -> tuple[list[str], dict[str, list[int]]]:
    """Group feature-matrix columns into interpretable buckets.

    Returns ``(order, groups)`` where ``order`` lists the bucket names (tabular features in
    canonical order, then ``visual_embedding`` if there is an embedding) and ``groups`` maps
    each bucket to the column indices of the assembled matrix it covers.
    """
    tab_names = [_to_original_name(n) for n in preprocessor.get_feature_names_out()]
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(tab_names):
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(i)
    if embedding_dim:
        start = len(tab_names)
        groups[VISUAL_FEATURE] = list(range(start, start + embedding_dim))
        order.append(VISUAL_FEATURE)
    return order, groups


def shap_values_per_class(
    bundle: ModelBundle, tabular: pd.DataFrame, embedding: np.ndarray | None
) -> np.ndarray:
    """Exact TreeSHAP values via XGBoost ``pred_contribs``.

    Returns an array of shape ``(n_rows, n_features, n_classes)`` — the per-feature
    contribution to each class margin (the bias term is dropped). Requires a tree model.
    """
    import xgboost as xgb

    if bundle.model_type != "xgboost":
        raise ValueError(
            f"native TreeSHAP needs an xgboost model, got {bundle.model_type!r}; "
            "explain via the shap library in the notebook instead."
        )
    emb = embedding if bundle.embedding_dim is not None else None
    matrix = assemble_matrix(tabular, emb, bundle.preprocessor)
    booster = bundle.model.get_booster()
    contribs = np.asarray(booster.predict(xgb.DMatrix(matrix), pred_contribs=True))

    if contribs.ndim == 3:
        # Multiclass: (n_rows, n_classes, n_features + 1) → drop bias, → (n, features, classes).
        return np.transpose(contribs[:, :, :-1], (0, 2, 1))
    # Binary / single output: (n_rows, n_features + 1) → (n, features, 1).
    return contribs[:, :-1][:, :, None]


def grouped_contributions(
    bundle: ModelBundle, tabular: pd.DataFrame, embedding: np.ndarray | None
) -> list[dict[str, float]]:
    """Per-row grouped SHAP for each row's predicted class.

    Returns one dict per row mapping bucket name → summed contribution. Summing within a
    bucket preserves the total (TreeSHAP additivity), so the visual bucket and the tabular
    features together still account for the full margin shift.
    """
    order, groups = build_feature_groups(bundle.preprocessor, bundle.embedding_dim)
    per_class = shap_values_per_class(bundle, tabular, embedding)
    _, proba = predict_frame(bundle, tabular, embedding)
    pred_idx = proba.argmax(axis=1)

    out: list[dict[str, float]] = []
    for row in range(per_class.shape[0]):
        row_vals = per_class[row, :, int(pred_idx[row])]
        out.append({name: float(row_vals[idxs].sum()) for name, idxs in groups.items()})
    return out


def explain_prediction(
    bundle: ModelBundle,
    tabular: pd.DataFrame,
    embedding: np.ndarray | None,
    top_k: int | None = None,
) -> dict[str, float]:
    """Explain a single prediction: bucket contributions sorted by magnitude (desc).

    Intended for the API: one row in, a ``{feature: contribution}`` dict out for the
    predicted class. ``top_k`` keeps only the strongest contributions.
    """
    grouped = grouped_contributions(bundle, tabular, embedding)[0]
    items = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)
    if top_k is not None:
        items = items[:top_k]
    return dict(items)


def grouped_shap_matrix(
    bundle: ModelBundle,
    tabular: pd.DataFrame,
    embedding: np.ndarray | None,
    class_index: int,
) -> tuple[np.ndarray, list[str]]:
    """Grouped SHAP for a fixed class across many rows — for the notebook summary plot.

    Returns ``(matrix, names)`` with ``matrix`` of shape ``(n_rows, n_buckets)`` and ``names``
    the bucket labels, ready to pass to ``shap.summary_plot``.
    """
    order, groups = build_feature_groups(bundle.preprocessor, bundle.embedding_dim)
    per_class = shap_values_per_class(bundle, tabular, embedding)
    vals = per_class[:, :, class_index]
    matrix = np.column_stack([vals[:, groups[name]].sum(axis=1) for name in order])
    return matrix, order
