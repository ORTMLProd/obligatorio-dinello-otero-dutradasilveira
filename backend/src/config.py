"""Application configuration via pydantic-settings.

Single source of config for the API. Values are resolved in priority order:
init args > environment variables (prefix ``API_``) > ``.env`` > ``configs/api.yaml``
> secrets. In containers the canonical source is environment variables (12-factor);
the YAML file is a convenience for local development and is therefore optional.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# This file is ``backend/src/config.py``. The monorepo root (where ``configs/`` lives)
# is three parents up. Inside the container this resolves to a path that does not
# exist, which is fine: the YAML source is optional (see ``settings_customise_sources``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT / "configs" / "api.yaml"


class Settings(BaseSettings):
    """Runtime settings for the API service."""

    service_name: str = "soccernet-events-api"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list[str] = []
    # Directory holding the exported clip model bundle (clip_model.pt). In Docker set
    # API_CLIP_MODEL_DIR to the mounted path.
    clip_model_dir: str = "models/clips-v1"

    def resolved_clip_model_dir(self) -> Path:
        path = Path(self.clip_model_dir)
        return path if path.is_absolute() else _REPO_ROOT / path

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``clip_model_dir`` uses the ``model_`` prefix pydantic reserves; opt out of the guard.
        protected_namespaces=(),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # YAML is optional: only add the source when the file actually exists, so the
        # app boots from env/defaults inside the container where ``configs/`` is absent.
        yaml_source: tuple[PydanticBaseSettingsSource, ...] = ()
        if _DEFAULT_YAML.is_file():
            yaml_source = (
                YamlConfigSettingsSource(
                    settings_cls, yaml_file=_DEFAULT_YAML, yaml_file_encoding="utf-8"
                ),
            )
        return (init_settings, env_settings, dotenv_settings, *yaml_source, file_secret_settings)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached ``Settings`` instance."""
    return Settings()
