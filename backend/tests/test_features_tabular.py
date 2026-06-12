"""Unit tests for the point-in-time tabular feature builder."""

from __future__ import annotations

from src.features.tabular import Annotation, build_tabular_features


def _ann(half: int, position_ms: int, label: str, team: str = "not applicable") -> Annotation:
    return Annotation(half=half, position_ms=position_ms, label=label, team=team, visible=True)


def test_minute_and_half_from_position() -> None:
    feats = build_tabular_features(
        annotations=[], half=1, position_ms=125_000, team="home", visible=True, league="england_epl"
    )
    assert feats["half"] == 1
    assert feats["minute"] == 2  # 125_000 ms // 60_000
    assert feats["league"] == "england_epl"
    assert feats["team_is_home"] == 1


def test_team_is_home_encoding() -> None:
    away = build_tabular_features([], 1, 0, team="away", visible=True, league="l")
    none = build_tabular_features([], 1, 0, team="not applicable", visible=False, league="l")
    assert away["team_is_home"] == 0
    assert none["team_is_home"] == -1
    assert none["visible"] == 0


def test_score_diff_accumulates_only_prior_goals() -> None:
    annotations = [
        _ann(1, 10_000, "Goal", team="home"),
        _ann(1, 20_000, "Goal", team="away"),
        _ann(2, 5_000, "Goal", team="home"),  # second half, after our query
    ]
    # Query mid first half, after the two first-half goals only would be t > 20_000.
    feats = build_tabular_features(
        annotations, half=1, position_ms=25_000, team="home", visible=True, league="l"
    )
    assert feats["score_diff"] == 0  # 1 home - 1 away, second-half goal excluded
    assert feats["events_so_far"] == 2


def test_secs_since_last_event() -> None:
    annotations = [_ann(1, 10_000, "Corner"), _ann(1, 30_000, "Foul")]
    feats = build_tabular_features(
        annotations, half=1, position_ms=33_000, team="home", visible=True, league="l"
    )
    assert feats["secs_since_last_event"] == 3.0  # 33s - 30s


def test_no_prior_event_sentinel() -> None:
    feats = build_tabular_features(
        [], half=1, position_ms=1_000, team="home", visible=True, league="l"
    )
    assert feats["events_so_far"] == 0
    assert feats["secs_since_last_event"] == -1.0
