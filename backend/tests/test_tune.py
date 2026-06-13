"""Tests for the Optuna tuning core (Phase 3 optimisation electivo).

The pure pieces are tested with ``FixedTrial`` (deterministic, no study needed). The
key course invariant — the search optimises *validation* only and never reads the test
split — is guarded by ``test_objective_never_reads_test_split``. The full ``run_tuning``
orchestration (MLflow, dataset) is verified live, like ``run()`` in train.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from optuna.trial import FixedTrial

from src.features.tabular import TABULAR_COLUMNS
from src.models.config import ModelSpec, SearchParamSpec, TuningConfig
from src.models.export import predict_frame
from src.models.train import SplitData
from src.models.tune import (
    _fit_bundle,
    objective,
    sample_params,
    sample_selected_columns,
    save_optuna_plots,
)

CLASSES = ["background", "card", "corner", "goal", "substitution"]
EMB_DIM = 8


def _synthetic_split(n_per_class: int, seed: int) -> SplitData:
    rng = np.random.default_rng(seed)
    rows, ys, embs = [], [], []
    for idx, _cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            rows.append(
                {
                    "half": int(rng.integers(1, 3)),
                    "minute": int(rng.integers(0, 50)),
                    "score_diff": int(rng.integers(-2, 3)),
                    "league": "england_epl",
                    "team_is_home": int(rng.integers(-1, 2)),
                    "visible": int(rng.integers(0, 2)),
                    "events_so_far": int(rng.integers(0, 20)),
                    "secs_since_last_event": float(rng.integers(-1, 300)),
                }
            )
            ys.append(idx)
            embs.append(rng.normal(loc=idx, scale=0.3, size=EMB_DIM))
    tabular = pd.DataFrame(rows, columns=list(TABULAR_COLUMNS))
    return SplitData(tabular=tabular, embedding=np.array(embs, dtype=np.float32), y=np.array(ys))


def test_sample_params_reads_search_space() -> None:
    """Each declared param is suggested with its configured type and returned by name."""
    space = {
        "max_depth": SearchParamSpec(type="int", low=3, high=10),
        "learning_rate": SearchParamSpec(type="float", low=0.01, high=0.3, log=True),
    }
    trial = FixedTrial({"max_depth": 6, "learning_rate": 0.1})
    params = sample_params(trial, space)
    assert params == {"max_depth": 6, "learning_rate": 0.1}


def test_sample_selected_columns_always_keeps() -> None:
    """With every toggle off, only the always-keep columns survive — in canonical order."""
    trial = FixedTrial({f"keep_{c}": False for c in TABULAR_COLUMNS if c != "league"})
    selected = sample_selected_columns(trial, always_keep=("league",))
    assert selected == ("league",)


def test_sample_selected_columns_preserves_canonical_order() -> None:
    """Kept columns follow TABULAR_COLUMNS order regardless of toggle order."""
    toggles = {f"keep_{c}": (c in ("minute", "half")) for c in TABULAR_COLUMNS if c != "league"}
    selected = sample_selected_columns(FixedTrial(toggles), always_keep=("league",))
    assert selected == ("half", "minute", "league")


def test_objective_returns_valid_f1() -> None:
    """The objective returns a validation macro-F1 in [0, 1]."""
    splits = {"train": _synthetic_split(20, seed=1), "val": _synthetic_split(5, seed=2)}
    cfg = TuningConfig(
        enabled=True,
        target_model="xgboost",
        select_features=True,
        always_keep=["league"],
        search_space={"max_depth": SearchParamSpec(type="int", low=2, high=4)},
    )
    trial = FixedTrial(
        {"max_depth": 3, **{f"keep_{c}": True for c in TABULAR_COLUMNS if c != "league"}}
    )
    score = objective(trial, splits, CLASSES, cfg, scale_numeric=False, use_embedding=True, seed=42)
    assert 0.0 <= score <= 1.0


def test_tuned_bundle_serves_with_full_request_contract() -> None:
    """Regression: a feature-selected bundle must still serve through the router path.

    The API builds the request frame from ``bundle.tabular_columns``. If feature selection
    shrank that list, ``assemble_matrix`` (which needs every TABULAR_COLUMNS) would KeyError
    at serving. ``tabular_columns`` must stay the full request contract; the dropped columns
    live only inside the fitted preprocessor.
    """
    train = _synthetic_split(20, seed=1)
    test = _synthetic_split(5, seed=2)
    splits = {"train": train, "test": test}
    selected = ("league", "half", "visible")  # a strict subset of the 8 tabular columns

    bundle = _fit_bundle(
        ModelSpec(type="xgboost", params={"max_depth": 3, "n_estimators": 30}),
        selected,
        splits,
        CLASSES,
        scale_numeric=False,
        use_embedding=True,
        seed=42,
        embedding_dim=EMB_DIM,
        version="v1-test",
        dataset_hash="deadbeef",
        config_hash="cafe",
    )

    # The request contract is the full column set, regardless of what the model uses.
    assert tuple(bundle.tabular_columns) == TABULAR_COLUMNS

    # Mimic the router: build the frame from bundle.tabular_columns, then serve.
    frame = test.tabular[list(bundle.tabular_columns)]
    labels, proba = predict_frame(bundle, frame, test.embedding)
    assert all(lbl in CLASSES for lbl in labels)


def test_save_optuna_plots_produces_files(tmp_path) -> None:
    """The viz helper renders the study plots to disk (best-effort, returns written paths)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0) + t.suggest_int("k", 1, 5), n_trials=8)

    paths = save_optuna_plots(study, tmp_path)
    assert len(paths) >= 1
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_objective_never_reads_test_split() -> None:
    """Anti-leakage: the search must score on validation only, never touching test.

    We pass a splits dict with no ``test`` key; if the objective read it, this would
    raise KeyError. Succeeding proves the test split is untouched during search.
    """
    splits = {"train": _synthetic_split(20, seed=1), "val": _synthetic_split(5, seed=2)}
    cfg = TuningConfig(
        enabled=True, search_space={"max_depth": SearchParamSpec(type="int", low=2, high=4)}
    )
    trial = FixedTrial(
        {"max_depth": 3, **{f"keep_{c}": True for c in TABULAR_COLUMNS if c != "league"}}
    )
    score = objective(trial, splits, CLASSES, cfg, scale_numeric=False, use_embedding=True, seed=42)
    assert 0.0 <= score <= 1.0
