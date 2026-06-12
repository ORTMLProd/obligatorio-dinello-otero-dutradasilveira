"""Pooling of pre-extracted ResNet features over a temporal window.

SoccerNet ships per-half ResNet feature arrays of shape ``(n_frames, dim)`` sampled
at a fixed fps. For a window centred on an annotated timestamp we pool the frames
within ``±half_window_ms`` into a single vector. This is the visual half of the
single preprocessing source (invariant 3); in Phase 3 it gains a frame-based path
for the fine-tuned CNN, but the pooled-feature contract stays the same.
"""

from __future__ import annotations

import numpy as np

_POOLERS = {
    "mean": lambda w: w.mean(axis=0),
    "max": lambda w: w.max(axis=0),
}


def pool_window(
    features: np.ndarray,
    position_ms: int,
    fps: int,
    half_window_ms: int,
    pool: str = "mean",
) -> np.ndarray:
    """Pool the feature rows within ``±half_window_ms`` of ``position_ms``.

    The window is clamped to the bounds of ``features`` so positions at or beyond the
    array edges still yield a finite vector (never an empty slice).

    Args:
        features: array of shape ``(n_frames, dim)`` for the relevant half.
        position_ms: window centre, in ms within the half.
        fps: sampling rate of ``features``.
        half_window_ms: half-width of the window, in ms.
        pool: aggregation strategy ("mean" or "max").

    Returns:
        A pooled vector of shape ``(dim,)``.
    """
    if pool not in _POOLERS:
        raise ValueError(f"unknown pool strategy {pool!r}; expected one of {sorted(_POOLERS)}")

    n_frames = len(features)
    center = round(position_ms / 1000 * fps)
    half_w = round(half_window_ms / 1000 * fps)

    lo = max(0, center - half_w)
    hi = min(n_frames, center + half_w + 1)
    if lo >= hi:
        # Centre is past the array: fall back to the single nearest frame.
        nearest = min(max(center, 0), n_frames - 1)
        lo, hi = nearest, nearest + 1

    return _POOLERS[pool](features[lo:hi])
