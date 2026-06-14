from src.models.clip_config import DEFAULT_CLIP_CONFIG_PATH, ClipTrainConfig


def test_clip_config_loads() -> None:
    cfg = ClipTrainConfig.from_yaml(DEFAULT_CLIP_CONFIG_PATH)
    assert cfg.backbone == "resnet18"
    assert cfg.k > 0 and cfg.frame_size > 0
    assert cfg.train.epochs > 0
    assert len(cfg.normalize.mean) == 3 and len(cfg.normalize.std) == 3
    assert cfg.mlflow.tracking_uri.startswith("sqlite") or cfg.mlflow.tracking_uri.startswith(
        "http"
    )


def test_clip_paths_resolved() -> None:
    cfg = ClipTrainConfig.from_yaml(DEFAULT_CLIP_CONFIG_PATH)
    assert cfg.paths.resolved("model_dir").name == "clips-v1"
