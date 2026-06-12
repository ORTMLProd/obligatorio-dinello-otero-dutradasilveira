"""Regression test for invariant 2: point-in-time correctness.

A tabular feature computed for a window at instant ``t`` must use ONLY information
available up to ``t`` — never future events. This is the canonical anti-leakage
test the course grades with most weight. We build a synthetic game and assert that
sliding the query time forward strictly reveals more (never less) past information,
and that future goals never leak into the score at ``t``.
"""

from __future__ import annotations

from src.features.tabular import Annotation, build_tabular_features


def _synthetic_game() -> list[Annotation]:
    return [
        Annotation(1, 5_000, "Corner", "home", True),
        Annotation(1, 15_000, "Goal", "home", True),  # home leads 1-0
        Annotation(1, 40_000, "Goal", "away", True),  # 1-1
        Annotation(2, 10_000, "Goal", "home", True),  # 2-1 (second half)
        Annotation(2, 30_000, "Substitution", "away", True),
    ]


def test_score_at_t_excludes_future_goals() -> None:
    game = _synthetic_game()
    # Just after the first goal: 1-0.
    assert build_tabular_features(game, 1, 16_000, "home", True, "l")["score_diff"] == 1
    # After the equaliser: 1-1.
    assert build_tabular_features(game, 1, 41_000, "away", True, "l")["score_diff"] == 0
    # Early second half before the third goal: still 1-1 (future goal excluded).
    assert build_tabular_features(game, 2, 5_000, "home", True, "l")["score_diff"] == 0
    # After the third goal: 2-1.
    assert build_tabular_features(game, 2, 11_000, "home", True, "l")["score_diff"] == 1


def test_events_so_far_is_monotonic_non_decreasing() -> None:
    game = _synthetic_game()
    queries = [(1, 0), (1, 10_000), (1, 20_000), (2, 0), (2, 20_000), (2, 60_000)]
    counts = [
        build_tabular_features(game, h, p, "home", True, "l")["events_so_far"] for h, p in queries
    ]
    assert counts == sorted(counts)
    assert counts[0] == 0  # nothing before kickoff
    assert counts[-1] == len(game)  # everything before end of game


def test_query_at_exact_event_time_excludes_that_event() -> None:
    """An event exactly at ``t`` is not yet observable at ``t`` (strictly-before)."""
    game = _synthetic_game()
    feats = build_tabular_features(game, 1, 15_000, "home", True, "l")
    assert feats["score_diff"] == 0  # the goal AT 15_000 does not count yet
    assert feats["events_so_far"] == 1  # only the corner at 5_000
