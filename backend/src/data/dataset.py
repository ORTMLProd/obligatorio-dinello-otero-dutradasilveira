"""Read the built dataset artifacts (manifest + pooled features).

A thin, reusable reader so consumers — the EDA notebook now, training in Phase 2 —
load the dataset the same way instead of re-implementing IO. The pooled feature
matrix is row-aligned with the manifest (row ``i`` ↔ ``window_id == i``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import DatasetConfig


def load_manifest(cfg: DatasetConfig) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.resolved("processed_dir") / "manifest.parquet")


def load_features(cfg: DatasetConfig) -> np.ndarray:
    return np.load(cfg.paths.resolved("processed_dir") / "resnet_pooled.npy")


def load_dataset(cfg: DatasetConfig) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(manifest, features)`` and assert they are row-aligned."""
    manifest = load_manifest(cfg)
    features = load_features(cfg)
    if len(manifest) != len(features):
        raise ValueError(
            f"manifest ({len(manifest)}) and features ({len(features)}) are misaligned"
        )
    return manifest, features
