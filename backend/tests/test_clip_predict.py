"""Tests for the /predict/clip endpoint (Fase 3.5)."""

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.clip_export import ClipModelMeta
from src.models.clip_model import build_clip_model

CLASSES = ["background", "card", "corner", "goal", "substitution"]


def _video_bytes(path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 48))
    for i in range(40):
        writer.write(np.full((48, 48, 3), i * 6 % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _inject_clip_model() -> None:
    app.state.clip_model = build_clip_model(len(CLASSES), hidden=32, pretrained=False)
    app.state.clip_meta = ClipModelMeta(
        backbone="resnet18",
        pooling="mean",
        classes=CLASSES,
        k=8,
        frame_size=32,
        hidden=32,
        dropout=0.3,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        model_version="clips-test",
        metrics={},
    )
    app.state.clip_device = torch.device("cpu")


def test_predict_clip_returns_prediction_and_gradcam(tmp_path) -> None:
    _inject_clip_model()
    try:
        data = _video_bytes(tmp_path / "clip.avi")
        resp = TestClient(app).post(
            "/predict/clip", files={"video": ("clip.avi", data, "video/x-msvideo")}
        )
    finally:
        app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_label"] in CLASSES
    assert len(body["gradcam"]) == 8
    assert set(body["probabilities"]) == set(CLASSES)


def test_predict_clip_503_when_no_model(tmp_path) -> None:
    app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
    data = _video_bytes(tmp_path / "clip.avi")
    resp = TestClient(app).post(
        "/predict/clip", files={"video": ("clip.avi", data, "video/x-msvideo")}
    )
    assert resp.status_code == 503


def test_predict_clip_batch_returns_aligned_predictions(tmp_path) -> None:
    _inject_clip_model()
    try:
        d1 = _video_bytes(tmp_path / "a.avi")
        d2 = _video_bytes(tmp_path / "b.avi")
        resp = TestClient(app).post(
            "/predict/clip/batch",
            files=[
                ("videos", ("a.avi", d1, "video/x-msvideo")),
                ("videos", ("b.avi", d2, "video/x-msvideo")),
            ],
        )
    finally:
        app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert [p["filename"] for p in preds] == ["a.avi", "b.avi"]
    for p in preds:
        assert p["predicted_label"] in CLASSES
        assert set(p["probabilities"]) == set(CLASSES)
        assert "gradcam" not in p  # batch omits Grad-CAM
