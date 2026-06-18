"""Tests for serving/clip_inference.py (Fase 3.5)."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import torch

from src.models.clip_export import ClipModelMeta
from src.models.clip_model import build_clip_model
from src.serving.clip_inference import frames_from_video, serve_clip


def _video_bytes(path, n_frames=40, fps=10, size=48) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), i * 6 % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_frames_from_video_returns_k_frames(tmp_path) -> None:
    data = _video_bytes(tmp_path / "clip.avi")
    frames = frames_from_video(data, k=8, frame_size=32, suffix=".avi")
    assert len(frames) == 8
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in frames)


def test_serve_clip_returns_prediction_and_overlays(tmp_path) -> None:
    classes = ["background", "card", "corner", "goal", "substitution"]
    meta = ClipModelMeta(
        backbone="resnet18",
        pooling="mean",
        classes=classes,
        k=8,
        frame_size=32,
        hidden=32,
        dropout=0.3,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        model_version="clips-test",
        metrics={},
    )
    model = build_clip_model(len(classes), hidden=32, pooling="mean", pretrained=False)
    data = _video_bytes(tmp_path / "clip.avi")

    label, proba, overlays = serve_clip(model, meta, data, torch.device("cpu"), suffix=".avi")
    assert label in classes
    assert abs(float(proba.sum()) - 1.0) < 1e-5
    assert len(overlays) == meta.k
    assert all(len(base64.b64decode(o)) > 0 for o in overlays)
