"""Extract evenly-spaced frames from a clip around a timestamp, using OpenCV.

Used by the clip dataset builder (and, later, by serving). Frames are read by seeking
to millisecond timestamps, so it works on long match videos without decoding the whole
file. No SoccerNet SDK dependency.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def clip_frame_timestamps_ms(center_ms: int, clip_ms: int, k: int, duration_ms: int) -> list[int]:
    """K evenly-spaced timestamps over ``[center-clip/2, center+clip/2]``, clamped to the video.

    The window is clamped to ``[0, duration_ms]`` so a clip near either end still yields K
    valid timestamps (collapsing toward the bound). ``k == 1`` returns the clip centre.
    """
    half = clip_ms // 2
    lo = max(0, center_ms - half)
    hi = min(duration_ms, center_ms + half)
    if k == 1:
        return [(lo + hi) // 2]
    step = (hi - lo) / (k - 1)
    return [int(round(lo + i * step)) for i in range(k)]


def video_duration_ms(video_path: Path) -> int:
    """Video duration in ms (frame count / fps). Raises if the file can't be opened."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(n_frames / max(fps, 1.0) * 1000)
    finally:
        cap.release()


def extract_clip_frames(
    video_path: Path, center_ms: int, clip_ms: int, k: int, size: int
) -> list[np.ndarray]:
    """Read K ``size×size`` RGB frames around ``center_ms`` from ``video_path``.

    Seeks by millisecond timestamp (the nearest decoded frame is used) and resizes to
    ``size×size``. Returns a list of K ``uint8`` arrays of shape ``(size, size, 3)`` in RGB
    order. If a seek lands past the last frame, the previous frame (or a black frame) is
    reused so the output always has exactly K frames. Raises ``FileNotFoundError`` if the
    video can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_ms = int(n_frames / max(fps, 1.0) * 1000)
        timestamps = clip_frame_timestamps_ms(center_ms, clip_ms, k, duration_ms)

        frames: list[np.ndarray] = []
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
            ok, frame = cap.read()
            if not ok:
                fallback = frames[-1] if frames else np.zeros((size, size, 3), dtype=np.uint8)
                frames.append(fallback.copy())
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (size, size)))
        return frames
    finally:
        cap.release()
