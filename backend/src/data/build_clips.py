"""Build the clip dataset: extract K frames per window and write clips_manifest.parquet.

Mirrors build_dataset.py but the visual half is real frames (not pooled ResNet features):
for each window (event + background) we extract K frames from the corresponding half video
and store their paths. Reuses the same windowing/labels/splits/tabular logic. Idempotent
(frames already on disk are kept) and seeded.

Usage:
    uv run python -m src.data.build_clips --config ../configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.frames import extract_clip_frames, video_duration_ms
from src.data.labels import LABELS_FILE, find_games, league_of, load_annotations
from src.data.splits import load_splits
from src.data.windows import event_windows, sample_background_positions
from src.features.tabular import build_tabular_features


def _save_frames(frames: list, out_dir: Path, processed_dir: Path) -> list[str]:
    """Write frames as JPGs; return their paths relative to ``processed_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rels: list[str] = []
    for i, frame in enumerate(frames):
        path = out_dir / f"frame_{i}.jpg"
        if not path.is_file():
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        rels.append(str(path.relative_to(processed_dir)))
    return rels


def build(cfg: DatasetConfig) -> pd.DataFrame:
    raw_dir = cfg.paths.resolved("raw_dir")
    processed = cfg.paths.resolved("processed_dir")
    frames_root = processed / "frames"
    splits = load_splits(cfg.paths.resolved("splits_file"))
    game_ids = find_games(raw_dir)
    if not game_ids:
        raise SystemExit("No games on disk; run src.data.download first.")

    rng = random.Random(cfg.seed)
    rows: list[dict] = []

    for game_id in game_ids:
        game_dir = raw_dir / game_id
        annotations = load_annotations(game_dir / LABELS_FILE)
        league = league_of(game_id)
        split = splits[game_id]

        half_videos: dict[int, Path] = {}
        half_durations: dict[int, int] = {}
        for half, fname in enumerate(cfg.clips.video_files, start=1):
            video_path = game_dir / fname
            if video_path.is_file():
                half_videos[half] = video_path
                half_durations[half] = video_duration_ms(video_path)

        events = event_windows(annotations, cfg.label_lookup())
        n_background = round(cfg.background.ratio * len(events))
        background = sample_background_positions(
            annotations=annotations,
            half_durations_ms=half_durations,
            min_gap_ms=int(cfg.background.min_gap_s * 1000),
            n_samples=n_background,
            rng=rng,
        )

        for window in events + background:
            video_path = half_videos.get(window.half)
            if video_path is None:
                continue  # window references a half we have no video for
            window_id = len(rows)
            frames = extract_clip_frames(
                video_path,
                window.position_ms,
                int(cfg.clips.clip_seconds * 1000),
                cfg.clips.k,
                cfg.clips.frame_size,
            )
            frame_paths = _save_frames(frames, frames_root / game_id / str(window_id), processed)
            tabular = build_tabular_features(
                annotations, window.half, window.position_ms, window.team, window.visible, league
            )
            rows.append(
                {
                    "window_id": window_id,
                    "game_id": game_id,
                    "split": split,
                    "label": window.label,
                    "position_ms": window.position_ms,
                    "frame_paths": frame_paths,
                    **tabular,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise SystemExit(
            "No clips extracted — ¿descargaste los videos? "
            "(clips.enabled=true + SOCCERNET_PASSWORD, luego src.data.download)."
        )
    _write_artifacts(cfg, manifest, n_games=len(game_ids))
    return manifest


def _write_artifacts(cfg: DatasetConfig, manifest: pd.DataFrame, n_games: int) -> None:
    processed = cfg.paths.resolved("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = processed / "clips_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    summary = {
        "seed": cfg.seed,
        "n_games": n_games,
        "n_clips": int(len(manifest)),
        "k": cfg.clips.k,
        "clip_seconds": cfg.clips.clip_seconds,
        "frame_size": cfg.clips.frame_size,
        "class_counts": {str(k): int(v) for k, v in Counter(manifest["label"]).items()},
        "split_counts": {str(k): int(v) for k, v in Counter(manifest["split"]).items()},
        "content_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    summary_path = cfg.paths.resolved("summary_file").parent / "clips_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"clips manifest: {manifest_path}  ({len(manifest)} clips)")
    print(f"classes:        {summary['class_counts']}")
    print(f"splits:         {summary['split_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    build(DatasetConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
