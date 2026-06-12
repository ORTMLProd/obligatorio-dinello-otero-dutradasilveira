"""Build the windowed dataset: manifest + pooled ResNet features + summary.

Pipeline (config-driven, idempotent, seeded):
    for each downloaded game →
        event windows (4 target classes) + sampled background windows
        → point-in-time tabular features (src/features/tabular)
        → pooled ResNet vector per window (src/features/visual)
        → split from configs/splits.yaml (invariant 1)
    → write manifest.parquet, resnet_pooled.npy (row-aligned), dataset_summary.json

The heavy artifacts (parquet, npy) live under gitignored ``data/processed/``; the
small summary (counts + content hash) is written to ``report/`` for traceability.

Usage:
    uv run python -m src.data.build_dataset --config ../configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.labels import find_games, league_of, load_annotations
from src.data.splits import load_splits
from src.data.windows import Window, event_windows, sample_background_positions
from src.features.tabular import build_tabular_features
from src.features.visual import pool_window

LABELS_FILE = "Labels-v2.json"


def _load_half_features(game_dir: Path, feature_files: list[str]) -> dict[int, np.ndarray]:
    """Load the per-half ResNet arrays, keyed by half number (1, 2, ...)."""
    half_features: dict[int, np.ndarray] = {}
    for half, fname in enumerate(feature_files, start=1):
        half_features[half] = np.load(game_dir / fname)
    return half_features


def _windows_for_game(
    cfg: DatasetConfig, annotations: list, half_features: dict[int, np.ndarray], rng: random.Random
) -> list[Window]:
    events = event_windows(annotations, cfg.label_lookup())
    # Half duration in ms is derived from the feature array length and its fps.
    half_durations_ms = {
        half: int(len(feats) / cfg.features.fps * 1000) for half, feats in half_features.items()
    }
    n_background = round(cfg.background.ratio * len(events))
    background = sample_background_positions(
        annotations=annotations,
        half_durations_ms=half_durations_ms,
        min_gap_ms=int(cfg.background.min_gap_s * 1000),
        n_samples=n_background,
        rng=rng,
    )
    return events + background


def build(cfg: DatasetConfig) -> pd.DataFrame:
    raw_dir = cfg.paths.resolved("raw_dir")
    splits = load_splits(cfg.paths.resolved("splits_file"))
    game_ids = find_games(raw_dir)
    if not game_ids:
        raise SystemExit("No games on disk; run src.data.download first.")

    rng = random.Random(cfg.seed)
    rows: list[dict] = []
    pooled: list[np.ndarray] = []

    for game_id in game_ids:
        game_dir = raw_dir / game_id
        annotations = load_annotations(game_dir / LABELS_FILE)
        half_features = _load_half_features(game_dir, cfg.features.files)
        league = league_of(game_id)
        split = splits[game_id]

        for w in _windows_for_game(cfg, annotations, half_features, rng):
            feats = half_features.get(w.half)
            if feats is None:
                continue  # annotation references a half we have no features for
            tabular = build_tabular_features(
                annotations, w.half, w.position_ms, w.team, w.visible, league
            )
            rows.append(
                {
                    "window_id": len(rows),
                    "game_id": game_id,
                    "split": split,
                    "label": w.label,
                    "position_ms": w.position_ms,
                    **tabular,
                }
            )
            pooled.append(
                pool_window(
                    feats,
                    w.position_ms,
                    cfg.features.fps,
                    cfg.window.half_window_ms,
                    cfg.features.pool,
                )
            )

    manifest = pd.DataFrame(rows)
    features = np.vstack(pooled).astype(np.float32)
    _write_artifacts(cfg, manifest, features, n_games=len(game_ids))
    return manifest


def _write_artifacts(
    cfg: DatasetConfig, manifest: pd.DataFrame, features: np.ndarray, n_games: int
) -> None:
    processed = cfg.paths.resolved("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / "manifest.parquet"
    features_path = processed / "resnet_pooled.npy"
    manifest.to_parquet(manifest_path, index=False)
    np.save(features_path, features)

    summary = {
        "seed": cfg.seed,
        "n_games": n_games,
        "n_windows": int(len(manifest)),
        "feature_dim": int(features.shape[1]),
        "class_counts": _counts(manifest["label"]),
        "split_counts": _counts(manifest["split"]),
        "split_class_counts": {
            split: _counts(group["label"]) for split, group in manifest.groupby("split")
        },
        "content_sha256": _hash_files(manifest_path, features_path),
    }
    summary_path = cfg.paths.resolved("summary_file")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"manifest:  {manifest_path}  ({len(manifest)} windows)")
    print(f"features:  {features_path}  {features.shape}")
    print(f"summary:   {summary_path}")
    print(f"classes:   {summary['class_counts']}")
    print(f"splits:    {summary['split_counts']}")


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(series).items()}


def _hash_files(*paths: Path) -> str:
    """Content hash over the generated artifacts for dataset traceability."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    build(DatasetConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
