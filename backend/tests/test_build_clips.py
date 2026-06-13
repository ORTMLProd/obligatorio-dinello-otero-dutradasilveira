import json

import cv2
import numpy as np
import yaml

from src.data.build_clips import build
from src.data.config import DatasetConfig


def _make_video(path, n_frames=40, fps=10, size=48) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), i * 6 % 256, dtype=np.uint8))
    writer.release()


def _make_game(raw_dir, game_id) -> None:
    gdir = raw_dir / game_id
    gdir.mkdir(parents=True, exist_ok=True)
    labels = {
        "annotations": [
            {
                "gameTime": "1 - 00:02",
                "position": "2000",
                "label": "Goal",
                "team": "home",
                "visibility": "visible",
            },
            {
                "gameTime": "2 - 00:02",
                "position": "2000",
                "label": "Corner",
                "team": "away",
                "visibility": "visible",
            },
        ]
    }
    (gdir / "Labels-v2.json").write_text(json.dumps(labels), encoding="utf-8")
    _make_video(gdir / "1_224p.mkv")
    _make_video(gdir / "2_224p.mkv")


def _config(tmp_path) -> DatasetConfig:
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    games = ["league_a/2020/match-1", "league_a/2020/match-2"]
    for g in games:
        _make_game(raw, g)
    splits = {games[0]: "train", games[1]: "test"}
    splits_file = tmp_path / "splits.yaml"
    splits_file.write_text(yaml.safe_dump(splits), encoding="utf-8")

    return DatasetConfig.model_validate(
        {
            "seed": 42,
            "num_games": 2,
            "source_split": "train",
            "game_ids": games,
            "target_labels": {"goal": ["Goal"], "corner": ["Corner"]},
            "window": {"half_window_ms": 2000},
            "background": {"ratio": 1, "min_gap_s": 5},
            "features": {"files": ["1_ResNET_TF2_PCA512.npy"], "fps": 2, "dim": 512},
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "paths": {
                "raw_dir": str(raw),
                "processed_dir": str(processed),
                "splits_file": str(splits_file),
                "summary_file": str(tmp_path / "summary.json"),
            },
            "clips": {"enabled": True, "k": 4, "clip_seconds": 2, "frame_size": 32},
        }
    )


def test_build_clips_manifest_has_k_existing_frames(tmp_path) -> None:
    cfg = _config(tmp_path)
    manifest = build(cfg)
    processed = cfg.paths.resolved("processed_dir")

    assert len(manifest) > 0
    assert {"window_id", "game_id", "split", "label", "frame_paths"} <= set(manifest.columns)
    for paths in manifest["frame_paths"]:
        assert len(paths) == cfg.clips.k
        for rel in paths:
            assert (processed / rel).is_file()


def test_build_clips_no_game_crosses_splits(tmp_path) -> None:
    """Invariante 1: ningún game_id aparece en más de un split."""
    manifest = build(_config(tmp_path))
    by_game = manifest.groupby("game_id")["split"].nunique()
    assert (by_game == 1).all()
