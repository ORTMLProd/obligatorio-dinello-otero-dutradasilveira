"""Typed loader for ``configs/train_clips.yaml`` (Fase 3.5 — CNN de clips)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# backend/src/models/clip_config.py → parents: [0]=models [1]=src [2]=backend [3]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group("name"), m.group("default") or "")

    return _ENV_PATTERN.sub(repl, value)


class HeadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)


class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    lr: float = Field(gt=0)
    batch_size: int = Field(gt=0)


class NormalizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: list[float] = Field(min_length=3, max_length=3)
    std: list[float] = Field(min_length=3, max_length=3)


class ClipMLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_uri: str
    experiment_name: str


class ClipPathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_dir: str
    metrics_dir: str
    manifest: str
    processed_dir: str

    def resolved(self, name: str) -> Path:
        return REPO_ROOT / getattr(self, name)


class ClipTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    backbone: str
    pooling: str
    k: int = Field(gt=0)
    frame_size: int = Field(gt=0)
    head: HeadConfig
    train: TrainParams
    augment: bool
    # Fine-tune the backbone's last block (layer4) instead of a fully frozen backbone.
    # ``finetune_lr`` is the (lower) LR applied to the unfrozen backbone params.
    finetune: bool = False
    finetune_lr: float = Field(default=1e-4, gt=0)
    finetune_blocks: list[str] = Field(default_factory=lambda: ["layer4"])
    normalize: NormalizeConfig
    mlflow: ClipMLflowConfig
    paths: ClipPathsConfig

    @classmethod
    def from_yaml(cls, path: Path | str) -> ClipTrainConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if isinstance(raw.get("mlflow", {}).get("tracking_uri"), str):
            raw["mlflow"]["tracking_uri"] = _expand_env(raw["mlflow"]["tracking_uri"])
        return cls.model_validate(raw)


DEFAULT_CLIP_CONFIG_PATH = REPO_ROOT / "configs" / "train_clips.yaml"
