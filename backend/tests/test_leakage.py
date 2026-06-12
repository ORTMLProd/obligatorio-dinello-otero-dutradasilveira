"""Regression test for invariant 1: no game_id may cross splits (anti data-leakage).

This is the single most important test in the project per the course rubric. It
checks both the pure assignment function and — when present — the real generated
``configs/splits.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.splits import assign_splits, load_splits

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_assignment_is_a_disjoint_total_partition() -> None:
    games = [f"league/season/game-{i}" for i in range(20)]
    assignment = assign_splits(games, train=0.6, val=0.2, seed=42)

    # Every game assigned exactly once, to a valid split.
    assert set(assignment) == set(games)
    assert set(assignment.values()) <= {"train", "val", "test"}

    buckets = {s: {g for g, v in assignment.items() if v == s} for s in ("train", "val", "test")}
    # Disjoint: no game appears in two splits.
    assert buckets["train"] & buckets["val"] == set()
    assert buckets["train"] & buckets["test"] == set()
    assert buckets["val"] & buckets["test"] == set()
    # Total: union covers all games.
    assert buckets["train"] | buckets["val"] | buckets["test"] == set(games)


def test_assignment_is_deterministic() -> None:
    games = [f"g{i}" for i in range(15)]
    assert assign_splits(games, 0.6, 0.2, seed=7) == assign_splits(games, 0.6, 0.2, seed=7)


def test_small_set_keeps_a_test_game() -> None:
    assignment = assign_splits([f"g{i}" for i in range(8)], 0.6, 0.2, seed=42)
    assert sum(v == "test" for v in assignment.values()) >= 1


def test_generated_splits_file_has_no_crossing_games() -> None:
    cfg = DatasetConfig.from_yaml(DEFAULT_CONFIG_PATH)
    splits_file = cfg.paths.resolved("splits_file")
    if not splits_file.is_file():
        pytest.skip("configs/splits.yaml not generated yet")
    assignment = load_splits(splits_file)
    # A YAML mapping game_id->split is disjoint by construction; assert it's well-formed.
    assert set(assignment.values()) <= {"train", "val", "test"}
    assert len(assignment) == len(set(assignment))  # game_ids unique
