"""Pure window-generation logic: event windows + background sampling.

Kept free of IO so it can be unit-tested in isolation (notably the background
sampling rule: every background window must sit ≥ ``min_gap`` from any annotation,
and the class is sub-sampled to a fixed ratio rather than its natural frequency).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.features.tabular import Annotation

BACKGROUND_LABEL = "background"
NOT_APPLICABLE = "not applicable"


@dataclass(frozen=True)
class Window:
    half: int
    position_ms: int
    label: str  # one of the target classes, or "background"
    team: str
    visible: bool


def event_windows(annotations: list[Annotation], label_lookup: dict[str, str]) -> list[Window]:
    """One window per annotation whose SoccerNet label maps to a target class."""
    windows: list[Window] = []
    for a in annotations:
        cls = label_lookup.get(a.label)
        if cls is not None:
            windows.append(Window(a.half, a.position_ms, cls, a.team, a.visible))
    return windows


def sample_background_positions(
    annotations: list[Annotation],
    half_durations_ms: dict[int, int],
    min_gap_ms: int,
    n_samples: int,
    rng: random.Random,
    step_ms: int = 1000,
) -> list[Window]:
    """Sample background windows ≥ ``min_gap_ms`` from any annotation.

    Candidates are placed on a regular grid within each half's duration and kept only
    if they are far enough from every annotation in that half (halves are temporally
    disjoint, so cross-half distance is irrelevant). The pooled candidates are then
    sub-sampled without replacement to ``n_samples``.
    """
    by_half: dict[int, list[int]] = {}
    for a in annotations:
        by_half.setdefault(a.half, []).append(a.position_ms)

    candidates: list[Window] = []
    for half, duration in half_durations_ms.items():
        events = by_half.get(half, [])
        for pos in range(0, max(duration, 1), step_ms):
            if all(abs(pos - e) >= min_gap_ms for e in events):
                candidates.append(Window(half, pos, BACKGROUND_LABEL, NOT_APPLICABLE, False))

    if n_samples >= len(candidates):
        return candidates
    return rng.sample(candidates, n_samples)
