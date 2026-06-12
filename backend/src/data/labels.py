"""Reading SoccerNet ``Labels-v2.json`` and locating downloaded games on disk.

Kept free of the SoccerNet SDK so the dataset builder and splitter can run without
the heavy ``data`` dependency group — they consume only what ``download.py`` left
on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.features.tabular import Annotation

LABELS_FILE = "Labels-v2.json"


def find_games(raw_dir: Path) -> list[str]:
    """Return the game_ids (paths relative to ``raw_dir``) that have a labels file.

    A game_id is the SoccerNet game path, e.g.
    ``england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley``. Sorted for
    determinism so downstream split assignment is reproducible.
    """
    return sorted(str(p.parent.relative_to(raw_dir)) for p in raw_dir.rglob(LABELS_FILE))


def load_annotations(labels_path: Path) -> list[Annotation]:
    """Parse a ``Labels-v2.json`` file into normalised annotations."""
    with open(labels_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    annotations: list[Annotation] = []
    for entry in payload.get("annotations", []):
        # gameTime looks like "1 - 12:34"; the half is the leading integer.
        half = int(entry["gameTime"].split(" - ", 1)[0])
        annotations.append(
            Annotation(
                half=half,
                position_ms=int(entry["position"]),
                label=entry["label"],
                team=entry.get("team", "not applicable"),
                visible=entry.get("visibility", "visible") == "visible",
            )
        )
    return annotations


def league_of(game_id: str) -> str:
    """The competition is the first path segment of the game_id."""
    return game_id.split("/", 1)[0]
