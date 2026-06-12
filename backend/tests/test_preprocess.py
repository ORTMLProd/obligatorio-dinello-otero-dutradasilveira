"""Tests for the shared preprocessor — the anti training-serving skew boundary (invariant 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import assemble_matrix, build_preprocessor
from src.features.tabular import TABULAR_COLUMNS


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(TABULAR_COLUMNS))


def _row(league: str = "england_epl", **overrides) -> dict:
    base = {
        "half": 1,
        "minute": 30,
        "score_diff": 1,
        "league": league,
        "team_is_home": 1,
        "visible": 1,
        "events_so_far": 5,
        "secs_since_last_event": 12.0,
    }
    base.update(overrides)
    return base


def test_serving_row_matches_training_batch() -> None:
    """A row transformed alone (serving) equals the same row inside a batch (training)."""
    train = _frame([_row(minute=10), _row(league="spain_laliga", minute=80), _row(minute=45)])
    pre = build_preprocessor(scale_numeric=True).fit(train[list(TABULAR_COLUMNS)])

    batch = assemble_matrix(train, embedding=None, preprocessor=pre)
    single = assemble_matrix(_frame([_row(minute=10)]), embedding=None, preprocessor=pre)

    np.testing.assert_allclose(single[0], batch[0])


def test_unknown_league_does_not_crash() -> None:
    """A league unseen at fit time yields an all-zero one-hot block, not an error."""
    train = _frame([_row(league="england_epl"), _row(league="spain_laliga")])
    pre = build_preprocessor(scale_numeric=False).fit(train[list(TABULAR_COLUMNS)])

    out = assemble_matrix(_frame([_row(league="brazil_serie_a")]), embedding=None, preprocessor=pre)

    # The two one-hot columns (epl, laliga) are the first block and must be all zero.
    assert out.shape[0] == 1
    np.testing.assert_array_equal(out[0, :2], np.zeros(2))


def test_embedding_is_concatenated_after_tabular() -> None:
    train = _frame([_row(), _row(league="spain_laliga")])
    pre = build_preprocessor(scale_numeric=True).fit(train[list(TABULAR_COLUMNS)])
    tab_only = assemble_matrix(train, embedding=None, preprocessor=pre)

    emb = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    fused = assemble_matrix(train, embedding=emb, preprocessor=pre)

    assert fused.shape == (2, tab_only.shape[1] + 4)
    np.testing.assert_allclose(fused[:, : tab_only.shape[1]], tab_only)
    np.testing.assert_allclose(fused[:, tab_only.shape[1] :], emb)


def test_embedding_row_count_mismatch_raises() -> None:
    train = _frame([_row(), _row()])
    pre = build_preprocessor().fit(train[list(TABULAR_COLUMNS)])
    try:
        assemble_matrix(train, embedding=np.zeros((1, 4), dtype=np.float32), preprocessor=pre)
    except ValueError:
        return
    raise AssertionError("expected ValueError on row-count mismatch")
