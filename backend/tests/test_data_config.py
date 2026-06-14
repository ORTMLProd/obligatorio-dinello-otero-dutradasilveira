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
