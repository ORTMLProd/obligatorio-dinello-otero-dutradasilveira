"""Fitted preprocessing for the model — the anti-skew boundary (invariant 3).

The tabular preprocessor (one-hot encoder for ``league`` + optional scaler for the
continuous numerics) is fitted ONCE on the train split, serialized with the model, and
loaded by the API at serving time — never re-fitted. ``assemble_matrix`` is the SINGLE
place where the model's input matrix ``[tabular ⊕ embedding]`` is built, called identically
by training and serving. Any drift between the two would be training-serving skew, so it
lives here and nowhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.tabular import TABULAR_COLUMNS

# Partition of TABULAR_COLUMNS by how each feature is treated. The union is exactly
# TABULAR_COLUMNS (asserted below); this is the contract shared with training and serving.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("league",)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "minute",
    "score_diff",
    "events_so_far",
    "secs_since_last_event",
)
# Already-encoded small-cardinality integers — left untouched (one-hot/scaling adds nothing).
PASSTHROUGH_COLUMNS: tuple[str, ...] = ("half", "team_is_home", "visible")

assert set(CATEGORICAL_COLUMNS) | set(NUMERIC_COLUMNS) | set(PASSTHROUGH_COLUMNS) == set(
    TABULAR_COLUMNS
), "preprocess column partition must cover exactly TABULAR_COLUMNS"


def build_preprocessor(
    scale_numeric: bool = True,
    selected_columns: tuple[str, ...] | None = None,
) -> ColumnTransformer:
    """Build the (unfitted) tabular preprocessor.

    Args:
        scale_numeric: standard-scale the continuous numerics. Helps linear models
            (LogReg); tree models (XGBoost) are scale-invariant and ignore it.
        selected_columns: subset of ``TABULAR_COLUMNS`` to keep, for tabular feature
            selection (Phase 3 optimisation). ``None`` keeps all (the v0 behaviour).
            Whatever is selected is still routed through the same cat/num/passthrough
            partition, so this stays the single preprocessing source (invariant 3).

    Returns:
        A ColumnTransformer expecting a DataFrame with ``TABULAR_COLUMNS`` (unselected
        columns are dropped by ``remainder="drop"``). ``league`` is one-hot encoded with
        ``handle_unknown="ignore"`` so an unseen league at serving yields an all-zero
        block instead of crashing.
    """
    selected = set(TABULAR_COLUMNS if selected_columns is None else selected_columns)
    unknown = selected - set(TABULAR_COLUMNS)
    if unknown:
        raise ValueError(f"selected_columns outside TABULAR_COLUMNS: {sorted(unknown)}")

    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    # Intersect each block with the selection, preserving the declared order, and drop
    # empty blocks so the ColumnTransformer never receives a transformer with no columns.
    blocks = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ("num", numeric_step, NUMERIC_COLUMNS),
        ("pass", "passthrough", PASSTHROUGH_COLUMNS),
    ]
    transformers = [
        (name, step, cols)
        for name, step, columns in blocks
        if (cols := [c for c in columns if c in selected])
    ]
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def assemble_matrix(
    tabular: pd.DataFrame,
    embedding: np.ndarray | None,
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Build the model input matrix ``[tabular_transformed ⊕ embedding]``.

    The ONLY place the feature matrix is assembled (invariant 3). Training fits
    ``preprocessor`` on the train split and calls this; serving loads the fitted
    ``preprocessor`` and calls this with the same code path.

    Args:
        tabular: rows containing ``TABULAR_COLUMNS`` (manifest slice or a one-row request).
        embedding: row-aligned pooled ResNet features, or ``None`` for a tabular-only model.
        preprocessor: a fitted ColumnTransformer from ``build_preprocessor``.

    Returns:
        A 2-D float array with one row per input row.
    """
    tab = np.asarray(preprocessor.transform(tabular[list(TABULAR_COLUMNS)]), dtype=np.float32)
    if embedding is None:
        return tab
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if len(emb) != len(tab):
        raise ValueError(f"tabular ({len(tab)}) and embedding ({len(emb)}) row counts differ")
    return np.hstack([tab, emb])
