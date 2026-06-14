import numpy as np

from src.models.clip_export import ClipModelMeta, load_clip_bundle, predict_clip, save_clip_bundle
from src.models.clip_model import build_clip_model

CLASSES = ["background", "card", "corner", "goal", "substitution"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _meta() -> ClipModelMeta:
    return ClipModelMeta(
        backbone="resnet18",
        pooling="mean",
        classes=CLASSES,
        k=8,
        frame_size=32,
        hidden=32,
        dropout=0.3,
        normalize_mean=MEAN,
        normalize_std=STD,
        model_version="clips-test",
        metrics={},
    )


def test_save_load_predict_roundtrip(tmp_path) -> None:
    meta = _meta()
    model = build_clip_model(len(CLASSES), hidden=32, pooling="mean", pretrained=False)
    save_clip_bundle(model, meta, tmp_path)

    reloaded, meta2 = load_clip_bundle(tmp_path)
    assert meta2.classes == CLASSES and meta2.k == 8

    frames = [np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(8)]
    label, proba = predict_clip(reloaded, meta2, frames)
    assert label in CLASSES
    assert proba.shape == (5,)
    assert abs(float(proba.sum()) - 1.0) < 1e-5
