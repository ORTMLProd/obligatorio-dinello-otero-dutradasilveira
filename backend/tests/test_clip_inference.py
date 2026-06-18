"""Tests for serving/clip_inference.py (Fase 3.5)."""

from __future__ import annotations

import cv2
import numpy as np

from src.serving.clip_inference import frames_from_video


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
