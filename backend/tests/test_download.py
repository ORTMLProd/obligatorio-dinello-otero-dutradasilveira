import pytest

from src.data.config import DatasetConfig
from src.data.download import files_for_game, require_password


def _cfg(enabled: bool) -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "seed": 42,
            "num_games": 1,
            "source_split": "train",
            "game_ids": [],
            "target_labels": {"goal": ["Goal"]},
            "window": {"half_window_ms": 2000},
            "background": {"ratio": 2, "min_gap_s": 30},
            "features": {"files": ["1_ResNET_TF2_PCA512.npy"], "fps": 2, "dim": 512},
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "paths": {
                "raw_dir": "data/raw/soccernet",
                "processed_dir": "data/processed",
                "splits_file": "configs/splits.yaml",
                "summary_file": "report/dataset_summary.json",
            },
            "clips": {"enabled": enabled, "video_files": ["1_224p.mkv", "2_224p.mkv"]},
        }
    )


def test_files_for_game_excludes_videos_when_clips_disabled() -> None:
    files = files_for_game(_cfg(enabled=False))
    assert "Labels-v2.json" in files
    assert "1_ResNET_TF2_PCA512.npy" in files
    assert "1_224p.mkv" not in files


def test_files_for_game_includes_videos_when_clips_enabled() -> None:
    files = files_for_game(_cfg(enabled=True))
    assert "1_224p.mkv" in files and "2_224p.mkv" in files


def test_require_password_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("SOCCERNET_PASSWORD", "s3cret")
    assert require_password() == "s3cret"


def test_require_password_raises_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("SOCCERNET_PASSWORD", raising=False)
    with pytest.raises(RuntimeError):
        require_password()
