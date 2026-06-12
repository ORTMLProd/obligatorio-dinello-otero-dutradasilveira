"""Unit tests for ResNet feature-window pooling."""

from __future__ import annotations

import numpy as np

from src.features.visual import pool_window


def _ramp(n: int, dim: int = 4) -> np.ndarray:
    """Array whose row i is filled with the value i (easy to reason about means)."""
    return np.tile(np.arange(n, dtype=np.float32).reshape(-1, 1), (1, dim))


def test_mean_pool_centered_window() -> None:
    feats = _ramp(100)
    # position 10s @ 2fps → center frame 20; ±2s → ±4 frames → rows 16..24 inclusive.
    pooled = pool_window(feats, position_ms=10_000, fps=2, half_window_ms=2_000, pool="mean")
    assert pooled.shape == (4,)
    assert np.allclose(pooled, 20.0)  # mean of 16..24 is 20


def test_window_clamped_at_start() -> None:
    feats = _ramp(100)
    pooled = pool_window(feats, position_ms=0, fps=2, half_window_ms=2_000, pool="mean")
    # center 0, window rows 0..4 → mean 2.
    assert np.allclose(pooled, 2.0)


def test_window_clamped_at_end() -> None:
    feats = _ramp(50)  # last index 49
    # position 26s @ 2fps → center 52, beyond the array; window clamps to the tail.
    pooled = pool_window(feats, position_ms=26_000, fps=2, half_window_ms=2_000, pool="mean")
    assert pooled.shape == (4,)
    assert np.all(np.isfinite(pooled))


def test_output_dim_matches_input() -> None:
    feats = _ramp(30, dim=512)
    pooled = pool_window(feats, position_ms=5_000, fps=2, half_window_ms=2_000, pool="mean")
    assert pooled.shape == (512,)
