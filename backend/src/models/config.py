"""Typed loader for ``configs/train.yaml``.

Mirrors ``src/data/config.py``: the YAML is the single source of truth for the
baseline training run, parsed into validated pydantic models so the pipeline stays
free of magic numbers and fails fast on malformed configs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

# backend/src/models/config.py → parents: [0]=models [1]=src [2]=backend [3]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Matches ``${VAR}`` and ``${VAR:-default}`` so the tracking URI can come from the
# environment (Docker) while keeping a local file-store default outside containers.
_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def _expand_env(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return os.environ.get(m.group("name"), m.group("default") or "")

    return _ENV_PATTERN.sub(repl, value)


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    type: str  # "logistic_regression" | "xgboost"
    params: dict = {}


class TrainFeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    use_embedding: bool = True
    scale_numeric: bool = True


class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tracking_uri: str
    experiment_name: str
    register_model: bool = True


class SearchParamSpec(BaseModel):
    """One hyperparameter's Optuna search range, declared in ``train.yaml``.

    ``int``/``float`` map to ``suggest_int``/``suggest_float``. ``log`` samples on a log
    scale (useful for ``learning_rate``); ``step`` discretises a non-log range.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["int", "float"]
    low: float
    high: float
    step: float | None = None
    log: bool = False


class TuningConfig(BaseModel):
    """Optuna tuning + tabular feature selection (Phase 3 optimisation electivo).

    Disabled by default so the v0 ``train.yaml`` keeps validating unchanged. The search
    maximises *validation* macro-F1 only — the test split is never read during search
    (anti data-leakage). ``always_keep`` columns are never dropped by feature selection.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    n_trials: int = Field(default=30, ge=1)
    timeout_s: int | None = None
    target_model: str = "xgboost"
    select_features: bool = True
    always_keep: list[str] = Field(default_factory=lambda: ["league"])
    search_space: dict[str, SearchParamSpec] = Field(default_factory=dict)


class TrainPathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_dir: str
    metrics_dir: str

    def resolved(self, name: str) -> Path:
        """Resolve a configured path against the repo root."""
        return REPO_ROOT / getattr(self, name)


class TrainConfig(BaseModel):
    """Full validated view of ``configs/train.yaml``."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    models: list[ModelSpec] = Field(min_length=1)
    features: TrainFeaturesConfig
    mlflow: MLflowConfig
    paths: TrainPathsConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainConfig:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        # Expand ``${VAR:-default}`` in the tracking URI before validation.
        if isinstance(raw.get("mlflow", {}).get("tracking_uri"), str):
            raw["mlflow"]["tracking_uri"] = _expand_env(raw["mlflow"]["tracking_uri"])
        return cls.model_validate(raw)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
