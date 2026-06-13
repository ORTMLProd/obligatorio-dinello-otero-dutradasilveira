"""Serialize / load the trained model bundle, and run inference from it.

The bundle pairs the fitted estimator with the fitted preprocessor and the metadata
needed to reproduce predictions and trace the run (invariants 3 and 6). Training writes
it; the API loads it at startup and never re-fits anything. ``predict_frame`` is the
single shared inference path so training-time and serving-time predictions agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features.preprocess import assemble_matrix

BUNDLE_FILENAME = "model.joblib"


@dataclass
class ModelBundle:
    """Everything the API needs to serve and trace the model."""

    model: Any  # fitted sklearn-compatible estimator with predict_proba
    preprocessor: ColumnTransformer  # fitted on the train split only
    classes: list[str]  # class order matching predict_proba columns
    tabular_columns: list[str]
    embedding_dim: int | None  # None ⇒ tabular-only model
    model_type: str
    model_version: str
    dataset_hash: str
    train_config_hash: str
    metrics: dict  # test-split summary surfaced by /model-info
    # Class proportions in the TRAIN split — the reference distribution the API exposes as a
    # Prometheus gauge, so monitoring can compare live predictions against it (drift baseline).
    # Defaults to empty for backward compatibility with bundles/tests predating Fase 3.4.
    train_class_ratio: dict[str, float] = field(default_factory=dict)


def save_bundle(bundle: ModelBundle, model_dir: Path) -> Path:
    """Serialize ``bundle`` to ``model_dir/model.joblib`` and return the path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / BUNDLE_FILENAME
    joblib.dump(asdict(bundle), path)
    return path


def load_bundle(model_dir: Path) -> ModelBundle:
    """Load a bundle written by ``save_bundle``."""
    path = model_dir / BUNDLE_FILENAME
    return ModelBundle(**joblib.load(path))


def predict_frame(
    bundle: ModelBundle,
    tabular: pd.DataFrame,
    embedding: np.ndarray | None,
) -> tuple[list[str], np.ndarray]:
    """Run inference for a batch of rows.

    The single inference path shared by training (sanity checks) and the serving API,
    so both assemble features and read class probabilities identically (invariant 3).

    Args:
        tabular: rows with ``bundle.tabular_columns``.
        embedding: row-aligned pooled features, or ``None`` for a tabular-only bundle.

    Returns:
        ``(labels, probabilities)`` — predicted class per row and the full probability
        matrix (columns ordered as ``bundle.classes``).
    """
    emb = embedding if bundle.embedding_dim is not None else None
    matrix = assemble_matrix(tabular, emb, bundle.preprocessor)
    proba = np.asarray(bundle.model.predict_proba(matrix), dtype=np.float64)
    labels = [bundle.classes[i] for i in proba.argmax(axis=1)]
    return labels, proba
