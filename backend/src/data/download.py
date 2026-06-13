"""Download SoccerNet labels, ResNet features, and optionally 224p video files.

Labels and ResNet features require no NDA password. Video files (``*.mkv``) are
gated by ``cfg.clips.enabled`` and require the NDA password from the environment
variable ``SOCCERNET_PASSWORD`` (read via :func:`require_password`; never logged).

The set of games is taken from ``configs/dataset.yaml`` (explicit ``game_ids`` or
the first ``num_games`` of ``source_split``). Idempotent: files already on disk
are skipped.

Usage:
    uv run python -m src.data.download --config ../configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.labels import LABELS_FILE


def files_for_game(cfg: DatasetConfig) -> list[str]:
    """Per-game files to download: labels + ResNet features, plus videos if clips enabled."""
    files = [LABELS_FILE, *cfg.features.files]
    if cfg.clips.enabled:
        files += list(cfg.clips.video_files)
    return files


def require_password() -> str:
    """Return the NDA password from the environment, or raise (never prints it)."""
    password = os.environ.get("SOCCERNET_PASSWORD")
    if not password:
        raise RuntimeError(
            "SOCCERNET_PASSWORD no está seteada — requerida para descargar videos (NDA). "
            "Definila en .env o en el entorno."
        )
    return password


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
    files = files_for_game(cfg)
    downloader = SoccerNetDownloader(LocalDirectory=str(raw_dir))
    if cfg.clips.enabled:
        downloader.password = require_password()

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
