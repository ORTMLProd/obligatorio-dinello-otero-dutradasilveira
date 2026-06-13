"""Typed loader for ``configs/dataset.yaml``.

The YAML is the single source of truth for regenerating the windowed dataset
(download → splits → build). Parsing it into validated pydantic models keeps the
pipeline free of magic numbers and fails fast on malformed configs.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# backend/src/data/config.py → parents: [0]=data [1]=src [2]=backend [3]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


class WindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    half_window_ms: int = Field(gt=0)


class BackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ratio: float = Field(gt=0)
    min_gap_s: float = Field(gt=0)


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    files: list[str] = Field(min_length=1)
    fps: int = Field(gt=0)
    dim: int = Field(gt=0)
    pool: str = "mean"


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = Field(gt=0, lt=1)
    val: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw_dir: str
    processed_dir: str
    splits_file: str
    summary_file: str

    def resolved(self, name: str) -> Path:
        """Resolve a configured path against the repo root."""
        return REPO_ROOT / getattr(self, name)


class ClipsConfig(BaseModel):
    """Clip dataset spec (Fase 3.5): K frames por ventana extraídos de los videos 224p."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    k: int = Field(default=8, gt=0)
    clip_seconds: float = Field(default=8.0, gt=0)
    frame_size: int = Field(default=224, gt=0)
    video_files: list[str] = Field(default_factory=lambda: ["1_224p.mkv", "2_224p.mkv"])


class DatasetConfig(BaseModel):
    """Full validated view of ``configs/dataset.yaml``."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    num_games: int = Field(gt=0)
    source_split: str
    game_ids: list[str] = []
    target_labels: dict[str, list[str]]
    window: WindowConfig
    background: BackgroundConfig
    features: FeaturesConfig
    split: SplitConfig
    paths: PathsConfig
    clips: ClipsConfig = Field(default_factory=ClipsConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> DatasetConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)

    def label_lookup(self) -> dict[str, str]:
        """Invert ``target_labels`` into ``{soccernet_label: our_class}``."""
        return {sn: cls for cls, sns in self.target_labels.items() for sn in sns}


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "dataset.yaml"
