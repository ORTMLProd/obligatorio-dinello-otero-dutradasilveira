from src.config import get_settings


def test_settings_has_clip_model_dir() -> None:
    s = get_settings()
    assert s.clip_model_dir  # default no vacío
    assert s.resolved_clip_model_dir().name == "clips-v1"
