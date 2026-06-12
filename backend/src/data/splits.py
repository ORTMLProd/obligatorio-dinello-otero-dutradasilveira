"""Assign games to train/val/test splits — ALWAYS by ``game_id`` (invariant 1).

Splitting by game (never by window or frame) is what prevents data leakage: every
window of a game lands in exactly one split. The assignment is deterministic (seeded)
and written to ``configs/splits.yaml`` so it is versioned in git — the leakage
contract lives there, not in regenerable data artifacts.

Usage:
    uv run python -m src.data.splits --config ../configs/dataset.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from src.data.config import DEFAULT_CONFIG_PATH, DatasetConfig
from src.data.labels import find_games


def assign_splits(game_ids: list[str], train: float, val: float, seed: int) -> dict[str, str]:
    """Partition game_ids into {train, val, test} disjointly and deterministically.

    The remaining fraction after train+val goes to test. With few games we round and
    let test absorb the remainder, guaranteeing every game gets exactly one split.
    """
    games = sorted(game_ids)  # stable input order before seeded shuffle
    rng = random.Random(seed)
    rng.shuffle(games)

    n = len(games)
    n_train = round(n * train)
    n_val = round(n * val)
    # Keep at least one game in test when there is more than one game.
    if n > 1 and n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    assignment: dict[str, str] = {}
    for i, game in enumerate(games):
        if i < n_train:
            assignment[game] = "train"
        elif i < n_train + n_val:
            assignment[game] = "val"
        else:
            assignment[game] = "test"
    return assignment


def write_splits(assignment: dict[str, str], path: Path) -> None:
    """Write the game_id → split mapping as sorted YAML (stable diffs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {g: assignment[g] for g in sorted(assignment)}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Generado por src/data/splits.py — NO editar a mano.\n")
        fh.write("# Asignacion de splits por game_id (invariante 1, anti data-leakage).\n")
        yaml.safe_dump(ordered, fh, allow_unicode=True, sort_keys=False)


def load_splits(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def generate(cfg: DatasetConfig) -> dict[str, str]:
    game_ids = find_games(cfg.paths.resolved("raw_dir"))
    if not game_ids:
        raise SystemExit("No games found on disk; run src.data.download first.")
    assignment = assign_splits(game_ids, cfg.split.train, cfg.split.val, cfg.seed)
    write_splits(assignment, cfg.paths.resolved("splits_file"))
    counts = {s: sum(v == s for v in assignment.values()) for s in ("train", "val", "test")}
    print(f"Wrote splits for {len(assignment)} games: {counts}")
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    generate(DatasetConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
