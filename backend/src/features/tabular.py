"""Point-in-time tabular features for a window (invariant 2, anti data-leakage).

A window at instant ``t`` may only use information available up to ``t``. Every
aggregate here (accumulated score, event counts, time since last event) filters
the game's annotations to those *strictly before* ``t``. This module is the single
source of truth for tabular preprocessing (invariant 3) and is imported by both the
dataset builder and — from Phase 2 on — the serving API.
"""

from __future__ import annotations

from dataclasses import dataclass

# Halves have variable real length. To order events across halves on a single axis
# we offset the second half by a constant larger than any plausible half duration.
HALF_OFFSET_MS = 60 * 60 * 1000  # 60 minutes

# SoccerNet labels that count as a goal for the running score.
GOAL_LABELS = frozenset({"Goal"})

NO_PRIOR_EVENT_SENTINEL = -1.0


@dataclass(frozen=True)
class Annotation:
    """One SoccerNet annotation, normalised from ``Labels-v2.json``."""

    half: int
    position_ms: int
    label: str
    team: str  # "home" | "away" | "not applicable"
    visible: bool


def absolute_ms(half: int, position_ms: int) -> int:
    """Map a (half, position) pair to a single monotonic timeline in milliseconds."""
    return (half - 1) * HALF_OFFSET_MS + position_ms


def build_tabular_features(
    annotations: list[Annotation],
    half: int,
    position_ms: int,
    team: str,
    visible: bool,
    league: str,
) -> dict[str, float | int | str]:
    """Compute point-in-time tabular features for a window.

    Args:
        annotations: all annotations of the game the window belongs to.
        half, position_ms: location of the window within the game.
        team: annotated team of the window ("not applicable" for background).
        visible: whether the annotated action is visible on screen.
        league: competition identifier (first path segment of the game).

    Returns:
        A flat dict of features. Only information strictly before ``t`` is used.
    """
    t = absolute_ms(half, position_ms)
    prior = [a for a in annotations if absolute_ms(a.half, a.position_ms) < t]

    home_goals = sum(1 for a in prior if a.label in GOAL_LABELS and a.team == "home")
    away_goals = sum(1 for a in prior if a.label in GOAL_LABELS and a.team == "away")

    last_event = max((absolute_ms(a.half, a.position_ms) for a in prior), default=None)
    secs_since_last = (t - last_event) / 1000 if last_event is not None else NO_PRIOR_EVENT_SENTINEL

    return {
        "half": half,
        "minute": position_ms // 60_000,
        "score_diff": home_goals - away_goals,
        "league": league,
        "team_is_home": 1 if team == "home" else (0 if team == "away" else -1),
        "visible": int(visible),
        "events_so_far": len(prior),
        "secs_since_last_event": secs_since_last,
    }


# Ordered tabular feature columns (the contract shared with training and serving).
TABULAR_COLUMNS: tuple[str, ...] = (
    "half",
    "minute",
    "score_diff",
    "league",
    "team_is_home",
    "visible",
    "events_so_far",
    "secs_since_last_event",
)
