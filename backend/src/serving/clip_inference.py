"""Serving the clip model on uploaded videos (Fase 3.5).

Extracts K frames from the uploaded video the SAME way as training (reusing
``extract_clip_frames``), runs the clip model and Grad-CAM, and returns the prediction plus
base64 JPG overlays. Frames are processed in memory; the temp video file is deleted (no
images are persisted — data policy).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np

from src.data.frames import extract_clip_frames, video_duration_ms


def frames_from_video(
    video_bytes: bytes, k: int, frame_size: int, suffix: str = ".mp4"
) -> list[np.ndarray]:
    """Extract K evenly-spaced frames spanning the whole uploaded clip. Deletes the temp file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(video_bytes)
        tmp_path = Path(handle.name)
    try:
        duration_ms = video_duration_ms(tmp_path)
        return extract_clip_frames(tmp_path, duration_ms // 2, duration_ms, k, frame_size)
    finally:
        tmp_path.unlink(missing_ok=True)


def _jpg_base64(frame_rgb: np.ndarray) -> str:
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("failed to encode overlay as JPG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
