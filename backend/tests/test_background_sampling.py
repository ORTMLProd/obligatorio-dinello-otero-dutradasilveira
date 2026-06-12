"""Unit tests for window generation and background sampling."""

from __future__ import annotations

import random

from src.data.windows import (
    BACKGROUND_LABEL,
    event_windows,
    sample_background_positions,
)
from src.features.tabular import Annotation


def test_event_windows_map_only_target_labels() -> None:
    anns = [
        Annotation(1, 1_000, "Goal", "home", True),
        Annotation(1, 2_000, "Ball out of play", "away", True),  # not a target
        Annotation(1, 3_000, "Yellow card", "away", True),
    ]
    lookup = {"Goal": "goal", "Yellow card": "card"}
    windows = event_windows(anns, lookup)
    assert [w.label for w in windows] == ["goal", "card"]


def test_background_windows_respect_min_gap() -> None:
    anns = [Annotation(1, 60_000, "Goal", "home", True)]
    bg = sample_background_positions(
        annotations=anns,
        half_durations_ms={1: 300_000},
        min_gap_ms=30_000,
        n_samples=1000,  # more than candidates → take all
        rng=random.Random(0),
    )
    assert bg, "should produce background candidates"
    for w in bg:
        assert w.label == BACKGROUND_LABEL
        assert abs(w.position_ms - 60_000) >= 30_000  # ≥30s from the only annotation


def test_background_sampling_respects_count_and_is_deterministic() -> None:
    anns = [Annotation(1, 10_000, "Corner", "home", True)]
    kwargs = dict(
        annotations=anns,
        half_durations_ms={1: 600_000},
        min_gap_ms=30_000,
        n_samples=5,
        rng=random.Random(123),
    )
    first = sample_background_positions(**{**kwargs, "rng": random.Random(123)})
    second = sample_background_positions(**{**kwargs, "rng": random.Random(123)})
    assert len(first) == 5
    assert [(w.half, w.position_ms) for w in first] == [(w.half, w.position_ms) for w in second]


def test_background_only_in_existing_halves() -> None:
    anns = [Annotation(2, 5_000, "Goal", "home", True)]
    bg = sample_background_positions(
        annotations=anns,
        half_durations_ms={1: 200_000, 2: 200_000},
        min_gap_ms=30_000,
        n_samples=1000,
        rng=random.Random(0),
    )
    assert {w.half for w in bg} <= {1, 2}
