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
import torch

from src.data.frames import extract_clip_frames, video_duration_ms
from src.models.clip_export import ClipModelMeta, predict_clip
from src.models.clip_gradcam import gradcam_clip, overlay_heatmap
from src.models.clip_model import build_transforms


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


def classify_clip(
    model,
    meta: ClipModelMeta,
    video_bytes: bytes,
    device: torch.device | None,
    suffix: str = ".mp4",
):
    """Lightweight serving path: video → frames → ``(label, proba)`` (no Grad-CAM).

    Used by ``/predict/clip/batch``: computing Grad-CAM overlays per frame for every clip in a
    batch would be prohibitively heavy, so batch only returns the class + probabilities.
    """
    frames = frames_from_video(video_bytes, meta.k, meta.frame_size, suffix=suffix)
    return predict_clip(model, meta, frames, device)


def serve_clip(
    model,
    meta: ClipModelMeta,
    video_bytes: bytes,
    device: torch.device | None,
    suffix: str = ".mp4",
):
    """Full serving path: video → frames → prediction + per-frame Grad-CAM overlays.

    Returns ``(label, proba, overlays)`` where ``overlays`` is a list of base64 JPG strings,
    one per frame, of the Grad-CAM heatmap blended on that frame.
    """
    frames = frames_from_video(video_bytes, meta.k, meta.frame_size, suffix=suffix)
    label, proba = predict_clip(model, meta, frames, device)

    # Build the clip tensor with the same eval transform (anti-skew) for Grad-CAM.
    transform = build_transforms(False, meta.frame_size, meta.normalize_mean, meta.normalize_std)
    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0)
    if device is not None:
        clip = clip.to(device)
    heatmaps, _ = gradcam_clip(model, clip, class_index=meta.classes.index(label))

    overlays = [_jpg_base64(overlay_heatmap(frames[i], heatmaps[i])) for i in range(len(frames))]
    return label, proba, overlays
