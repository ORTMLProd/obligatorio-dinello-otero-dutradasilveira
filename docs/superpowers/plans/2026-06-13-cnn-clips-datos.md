# CNN de clips — sub-proyecto 1 (Datos) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Producir un dataset de clips (K frames por ventana, extraídos de los videos 224p de SoccerNet) con un `clips_manifest.parquet`, reusando el windowing/labels/splits existente.

**Architecture:** Se extiende `download.py` para bajar videos (password NDA del entorno), un módulo nuevo `frames.py` extrae K frames equiespaciados de un clip con OpenCV, y `build_clips.py` orquesta: por cada ventana (evento + background) extrae frames, los guarda (gitignored) y arma el manifest. Todo config-driven vía una sección `clips` nueva en `dataset.yaml`.

**Tech Stack:** Python 3.12, OpenCV (`opencv-python`), pandas/pyarrow, pydantic, SoccerNet SDK (descarga), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-cnn-clips-datos-design.md`

**Branch:** `feat/fase-3.5-datos` (ya creada; el spec ya está commiteado ahí).

---

## Convenciones del repo (recordatorio)

- Correr todo desde `backend/` con `uv run ...`. Tests: `uv run pytest`. Lint: `uv run ruff check . && uv run ruff format --check .`.
- Código/identificadores/docstrings en **inglés**; mensajes de commit en **español**, conventional commits, **sin** firma de Claude.
- `data/` y las imágenes (`*.jpg/*.png`) están gitignored: los frames y videos **nunca** se commitean.

---

## File Structure

- Modify: `backend/pyproject.toml` — agregar `opencv-python` al grupo `data`.
- Modify: `configs/dataset.yaml` — `num_games: 16` + sección `clips`.
- Modify: `backend/src/data/config.py` — `ClipsConfig` + campo `clips` en `DatasetConfig`.
- Create: `backend/src/data/frames.py` — `clip_frame_timestamps_ms`, `video_duration_ms`, `extract_clip_frames`.
- Create: `backend/tests/test_frames.py`.
- Modify: `backend/src/data/download.py` — `files_for_game`, `require_password`, descarga de videos.
- Create: `backend/src/data/build_clips.py` — builder del clips_manifest.
- Create: `backend/tests/test_build_clips.py`.

---

## Task 1: Dependencia OpenCV + config `clips`

**Files:**
- Modify: `backend/pyproject.toml` (grupo `data`)
- Modify: `backend/src/data/config.py`
- Modify: `configs/dataset.yaml`
- Test: `backend/tests/test_data_config.py` (crear si no existe; si existe, agregar)

- [ ] **Step 1: Write the failing test**

Crear/abrir `backend/tests/test_data_config.py` y agregar:

```python
from src.data.config import DEFAULT_CONFIG_PATH, ClipsConfig, DatasetConfig


def test_dataset_config_has_clips_section() -> None:
    cfg = DatasetConfig.from_yaml(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg.clips, ClipsConfig)
    assert cfg.clips.k > 0
    assert cfg.clips.frame_size > 0
    assert cfg.clips.video_files  # no vacío


def test_clips_defaults_disabled() -> None:
    # Sin sección clips, el default es deshabilitado (compat hacia atrás).
    c = ClipsConfig()
    assert c.enabled is False
    assert c.k == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_data_config.py -q`
Expected: FAIL con `ImportError: cannot import name 'ClipsConfig'`.

- [ ] **Step 3: Add `opencv-python` to the data group**

En `backend/pyproject.toml`, en el grupo `data`:

```toml
data = [
    "SoccerNet>=0.1.60",
    # Extracción de frames de los videos para el dataset de clips (Fase 3.5).
    "opencv-python>=4.10",
]
```

Luego: `cd backend && uv sync --group data`

- [ ] **Step 4: Add `ClipsConfig` and wire it into `DatasetConfig`**

En `backend/src/data/config.py`, agregar antes de `class DatasetConfig`:

```python
class ClipsConfig(BaseModel):
    """Clip dataset spec (Fase 3.5): K frames por ventana extraídos de los videos 224p."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    k: int = Field(default=8, gt=0)
    clip_seconds: float = Field(default=8.0, gt=0)
    frame_size: int = Field(default=224, gt=0)
    video_files: list[str] = Field(default_factory=lambda: ["1_224p.mkv", "2_224p.mkv"])
```

Y dentro de `class DatasetConfig`, agregar el campo (después de `paths`):

```python
    clips: ClipsConfig = Field(default_factory=ClipsConfig)
```

- [ ] **Step 5: Add the `clips` section to `configs/dataset.yaml`**

Cambiar `num_games: 8` → `num_games: 16` y agregar al final (antes de `paths:` o después, en el nivel raíz):

```yaml
# Dataset de clips (Fase 3.5): K frames por ventana extraídos de los videos 224p.
# Requiere SOCCERNET_PASSWORD para descargar videos (política NDA).
clips:
  enabled: true
  k: 8 # frames por clip
  clip_seconds: 8 # ventana temporal del clip (±4s alrededor del evento)
  frame_size: 224 # lado del frame (cuadrado), apto para ResNet
  video_files: ["1_224p.mkv", "2_224p.mkv"] # un video por mitad
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_data_config.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/data/config.py configs/dataset.yaml backend/tests/test_data_config.py
git commit -m "feat: config de clips (OpenCV + seccion clips en dataset.yaml)"
```

---

## Task 2: `clip_frame_timestamps_ms` (función pura)

**Files:**
- Create: `backend/src/data/frames.py`
- Test: `backend/tests/test_frames.py`

- [ ] **Step 1: Write the failing test**

Crear `backend/tests/test_frames.py`:

```python
from src.data.frames import clip_frame_timestamps_ms


def test_timestamps_are_evenly_spaced_within_clip() -> None:
    ts = clip_frame_timestamps_ms(center_ms=10_000, clip_ms=8_000, k=5, duration_ms=100_000)
    # clip [6000, 14000], 5 puntos equiespaciados.
    assert ts == [6000, 8000, 10000, 12000, 14000]


def test_timestamps_clamped_to_video_bounds() -> None:
    # center cerca del inicio: el límite inferior se clampea a 0.
    ts = clip_frame_timestamps_ms(center_ms=1_000, clip_ms=8_000, k=3, duration_ms=100_000)
    assert ts[0] == 0
    assert ts[-1] == 5000  # 1000 + 4000
    # center cerca del final: el superior se clampea a duration.
    ts2 = clip_frame_timestamps_ms(center_ms=99_000, clip_ms=8_000, k=3, duration_ms=100_000)
    assert ts2[-1] == 100_000


def test_single_frame_returns_center() -> None:
    assert clip_frame_timestamps_ms(center_ms=5_000, clip_ms=8_000, k=1, duration_ms=100_000) == [5000]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_frames.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.frames'`.

- [ ] **Step 3: Write minimal implementation**

Crear `backend/src/data/frames.py`:

```python
"""Extract evenly-spaced frames from a clip around a timestamp, using OpenCV.

Used by the clip dataset builder (and, later, by serving). Frames are read by seeking
to millisecond timestamps, so it works on long match videos without decoding the whole
file. No SoccerNet SDK dependency.
"""

from __future__ import annotations

from pathlib import Path


def clip_frame_timestamps_ms(center_ms: int, clip_ms: int, k: int, duration_ms: int) -> list[int]:
    """K evenly-spaced timestamps over ``[center-clip/2, center+clip/2]``, clamped to the video.

    The window is clamped to ``[0, duration_ms]`` so a clip near either end still yields K
    valid timestamps (collapsing toward the bound). ``k == 1`` returns the clip centre.
    """
    half = clip_ms // 2
    lo = max(0, center_ms - half)
    hi = min(duration_ms, center_ms + half)
    if k == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (k - 1)
    return [int(round(lo + i * step)) for i in range(k)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_frames.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/data/frames.py backend/tests/test_frames.py
git commit -m "feat: timestamps equiespaciados de clip (frames.py)"
```

---

## Task 3: `video_duration_ms` y `extract_clip_frames` (OpenCV)

**Files:**
- Modify: `backend/src/data/frames.py`
- Test: `backend/tests/test_frames.py`

- [ ] **Step 1: Write the failing test**

Agregar a `backend/tests/test_frames.py` (arriba, los imports y un helper para crear un video sintético):

```python
import cv2
import numpy as np

from src.data.frames import extract_clip_frames, video_duration_ms


def _make_video(path, n_frames=30, fps=10, size=64) -> None:
    """Escribe un .avi MJPG con n_frames cuadros de color distinto por índice."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        frame = np.full((size, size, 3), i * 8 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_duration_ms_matches_frames_over_fps(tmp_path) -> None:
    vp = tmp_path / "clip.avi"
    _make_video(vp, n_frames=30, fps=10)
    # 30 frames / 10 fps = 3s = 3000ms (tolerancia por redondeo del contenedor).
    assert abs(video_duration_ms(vp) - 3000) <= 200


def test_extract_clip_frames_returns_k_resized_rgb(tmp_path) -> None:
    vp = tmp_path / "clip.avi"
    _make_video(vp, n_frames=30, fps=10, size=64)
    frames = extract_clip_frames(vp, center_ms=1500, clip_ms=2000, k=8, size=32)
    assert len(frames) == 8
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in frames)


def test_extract_clip_frames_raises_on_missing_video(tmp_path) -> None:
    try:
        extract_clip_frames(tmp_path / "nope.avi", center_ms=0, clip_ms=2000, k=4, size=32)
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError on missing video")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_frames.py -q`
Expected: FAIL con `ImportError: cannot import name 'extract_clip_frames'`.

- [ ] **Step 3: Write minimal implementation**

Agregar a `backend/src/data/frames.py`:

```python
import cv2
import numpy as np


def video_duration_ms(video_path: Path) -> int:
    """Video duration in ms (frame count / fps). Raises if the file can't be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(n_frames / max(fps, 1.0) * 1000)
    finally:
        cap.release()


def extract_clip_frames(
    video_path: Path, center_ms: int, clip_ms: int, k: int, size: int
) -> list[np.ndarray]:
    """Read K ``size×size`` RGB frames around ``center_ms`` from ``video_path``.

    Seeks by millisecond timestamp (the nearest decoded frame is used) and resizes to
    ``size×size``. Returns a list of K ``uint8`` arrays of shape ``(size, size, 3)`` in RGB
    order. If a seek lands past the last frame, the previous frame (or a black frame) is
    reused so the output always has exactly K frames. Raises ``FileNotFoundError`` if the
    video can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_ms = int(n_frames / max(fps, 1.0) * 1000)
        timestamps = clip_frame_timestamps_ms(center_ms, clip_ms, k, duration_ms)

        frames: list[np.ndarray] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ok, frame = cap.read()
            if not ok:
                fallback = frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8)
                frames.append(fallback.copy())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (size, size)))
        return frames
    finally:
        cap.release()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_frames.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/data/frames.py backend/tests/test_frames.py
git commit -m "feat: extraccion de frames de clip con OpenCV (video_duration_ms, extract_clip_frames)"
```

---

## Task 4: Descarga de videos en `download.py`

**Files:**
- Modify: `backend/src/data/download.py`
- Test: `backend/tests/test_download.py` (crear)

- [ ] **Step 1: Write the failing test**

Crear `backend/tests/test_download.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_download.py -q`
Expected: FAIL con `ImportError: cannot import name 'files_for_game'`.

- [ ] **Step 3: Write minimal implementation**

En `backend/src/data/download.py`, agregar imports y dos helpers, y usar la password al descargar. Reemplazar el cuerpo de `download()` para incluir videos:

```python
import os

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
```

Y en `download()`, después de crear el `downloader` y antes del loop, setear la password sólo si se piden videos; y usar `files_for_game`:

```python
    files = files_for_game(cfg)
    downloader = SoccerNetDownloader(LocalDirectory=str(raw_dir))
    if cfg.clips.enabled:
        downloader.password = require_password()
```

(Eliminar la línea previa `files = [LABELS_FILE, *cfg.features.files]`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_download.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/data/download.py backend/tests/test_download.py
git commit -m "feat: descarga de videos con password NDA (files_for_game, require_password)"
```

---

## Task 5: `build_clips.py` — builder del manifest de clips

**Files:**
- Create: `backend/src/data/build_clips.py`
- Test: `backend/tests/test_build_clips.py`

- [ ] **Step 1: Write the failing test**

Crear `backend/tests/test_build_clips.py`. El fixture crea 2 "partidos" sintéticos (labels + dos videos por mitad), un `splits.yaml`, y un `dataset.yaml` apuntando a ese `raw_dir`; corre `build` y verifica manifest + leakage.

```python
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
            {"gameTime": "1 - 00:02", "position": "2000", "label": "Goal", "team": "home",
             "visibility": "visible"},
            {"gameTime": "2 - 00:02", "position": "2000", "label": "Corner", "team": "away",
             "visibility": "visible"},
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_build_clips.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.data.build_clips'`.

- [ ] **Step 3: Write minimal implementation**

Crear `backend/src/data/build_clips.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_build_clips.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run full suite + lint**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: todo verde (incluyendo los tests previos del proyecto). Si ruff format marca archivos, correr `uv run ruff format .` y volver a chequear.

- [ ] **Step 6: Commit**

```bash
git add backend/src/data/build_clips.py backend/tests/test_build_clips.py
git commit -m "feat: builder del dataset de clips (build_clips, clips_manifest.parquet)"
```

---

## Task 6: Correr el pipeline real (16 partidos) — requiere NDA + red

> Este paso descarga datos reales (videos NDA) y NO es un test automatizado. Requiere `SOCCERNET_PASSWORD` y conexión. Genera artefactos gitignored. Ejecutar localmente quien tenga la password.

- [ ] **Step 1: Setear la password (no commitear)**

Asegurar que existe `.env` (gitignored) con `SOCCERNET_PASSWORD=...`, o exportarla en el shell:

```bash
export SOCCERNET_PASSWORD='<password NDA>'
```

- [ ] **Step 2: Descargar labels + features + videos de los 16 partidos**

Run: `cd backend && uv run python -m src.data.download --config ../configs/dataset.yaml`
Expected: baja los 8 partidos nuevos (labels+features+videos) y los videos de los 8 ya presentes. Idempotente. Puede tardar (varios GB).

- [ ] **Step 3: Regenerar splits para 16 partidos**

Run: `cd backend && uv run python -m src.data.splits --config ../configs/dataset.yaml`
Expected: `Wrote splits for 16 games: {...}`. Revisar el diff de `configs/splits.yaml`.

- [ ] **Step 4: Construir el dataset de clips**

Run: `cd backend && uv run python -m src.data.build_clips --config ../configs/dataset.yaml`
Expected: imprime `clips manifest: .../clips_manifest.parquet (N clips)` con los conteos por clase/split. Verificar que `data/processed/frames/` tiene imágenes y `report/clips_summary.json` existe.

- [ ] **Step 5: (Consistencia) reconstruir dataset ResNet + reentrenar modelo de ventana para 16**

> Mantiene coherencia de splits entre el modelo de ventana y el de clips. Mecánico, comandos existentes.

```bash
cd backend
uv run python -m src.data.build_dataset --config ../configs/dataset.yaml
uv run python -m src.models.tune --config ../configs/train.yaml
```

Expected: el dataset ResNet y el modelo tuneado se regeneran sobre 16 partidos (mejora esperable de las clases minoritarias).

- [ ] **Step 6: Commit de artefactos versionables**

```bash
git add configs/splits.yaml report/clips_summary.json report/dataset_summary.json
git commit -m "data: splits y summaries regenerados para 16 partidos"
```

(Los videos, frames y parquets están gitignored — verificar con `git status` que NO aparezcan.)

---

## Self-Review (hecho)

- **Cobertura del spec:** descarga de videos (Task 4), `frames.py` (Tasks 2-3), `build_clips` + manifest (Task 5), config `clips` (Task 1), splits por game_id reusados (Task 5/6), NDA (Task 4/6), testing con video sintético + leakage (Tasks 3/5). ✓
- **Placeholders:** ninguno; todos los pasos con código/comando reales. ✓
- **Consistencia de tipos:** `clip_frame_timestamps_ms` / `extract_clip_frames` / `video_duration_ms` / `files_for_game` / `require_password` / `build` con firmas consistentes entre tasks y tests. ✓
- **Fuera de alcance:** entrenamiento multi-frame, Grad-CAM, serving, frontend — otros sub-proyectos.
