"""Download SoccerNet labels + pre-extracted ResNet features for a small subset.

Camino liviano (Fase 1): only ``Labels-v2.json`` and the per-half ResNet feature
arrays are fetched — no videos, so **no NDA password is required**. The set of
games is taken from ``configs/dataset.yaml`` (explicit ``game_ids`` or the first
``num_games`` of ``source_split``). Idempotent: files already on disk are skipped.

Usage:
    uv run python -m src.data.download --config ../configs/dataset.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.labels import LABELS_FILE


def resolve_game_ids(cfg: DatasetConfig) -> list[str]:
    """Return the games to download: explicit list, or first N of the source split."""
    if cfg.game_ids:
        return list(cfg.game_ids)
    from SoccerNet.utils import getListGames  # imported lazily (heavy, data group only)

    return list(getListGames(cfg.source_split))[: cfg.num_games]


def download(cfg: DatasetConfig) -> list[str]:
    """Download labels + features for the configured games. Returns the game_ids."""
    from SoccerNet.Downloader import SoccerNetDownloader

    raw_dir = cfg.paths.resolved("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)

    game_ids = resolve_game_ids(cfg)
    files = [LABELS_FILE, *cfg.features.files]
    downloader = SoccerNetDownloader(LocalDirectory=str(raw_dir))

    for game_id in game_ids:
        missing = [f for f in files if not (raw_dir / game_id / f).is_file()]
        if not missing:
            print(f"[skip] {game_id} (already complete)")
            continue
        print(f"[get ] {game_id} -> {missing}")
        downloader.downloadGame(game=game_id, files=missing, spl=cfg.source_split)

    print(f"Done. {len(game_ids)} games in {raw_dir}")
    return game_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    download(DatasetConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
