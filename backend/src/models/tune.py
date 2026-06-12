"""Hyperparameter tuning (Optuna) + tabular feature selection — Phase 3 optimisation.

Two sub-techniques of the *model optimisation* electivo, both with measured impact:

1. **Optuna search** over the target model's hyperparameters.
2. **Tabular feature selection**: each trial also picks a subset of ``TABULAR_COLUMNS``
   (``always_keep`` columns excepted), routed through the same selection-aware
   preprocessor — so this never duplicates preprocessing (invariant 3).

The objective maximises **validation** macro-F1 only; the test split is read exactly once
at the end, for the chosen model (selecting on test would be data-leakage, invariant 1/2).

We log a fair before/after to MLflow — a ``baseline`` run (default params, all features)
and a ``tuned-best`` run — plus inference **latency** (p50/p95), since the electivo
requires measuring impact on ML metrics *and* latency. The winning bundle is exported via
``save_bundle``, exactly like the baseline, so the API serves the optimised model.

Usage:
    uv run python -m src.models.tune --config ../configs/train.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import DEFAULT_CONFIG_PATH as DATASET_CONFIG_PATH
from src.data.config import DatasetConfig
from src.data.dataset import load_dataset
from src.features.preprocess import build_preprocessor
from src.features.tabular import TABULAR_COLUMNS
from src.models.config import DEFAULT_CONFIG_PATH as TRAIN_CONFIG_PATH
from src.models.config import ModelSpec, SearchParamSpec, TrainConfig, TuningConfig
from src.models.export import ModelBundle, predict_frame, save_bundle
from src.models.train import (
    SplitData,
    evaluate_on,
    fit_one,
    split_dataset,
)


def sample_params(trial, search_space: dict[str, SearchParamSpec]) -> dict:
    """Sample one hyperparameter dict from ``search_space`` for an Optuna ``trial``."""
    out: dict[str, float | int] = {}
    for name, spec in search_space.items():
        if spec.type == "int":
            if spec.log:
                out[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=True)
            else:
                step = int(spec.step) if spec.step is not None else 1
                out[name] = trial.suggest_int(name, int(spec.low), int(spec.high), step=step)
        else:
            if spec.step is not None:
                out[name] = trial.suggest_float(name, spec.low, spec.high, step=spec.step)
            else:
                out[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    return out


def sample_selected_columns(trial, always_keep: tuple[str, ...]) -> tuple[str, ...]:
    """Pick a subset of ``TABULAR_COLUMNS`` for this trial (feature selection).

    ``always_keep`` columns are never toggled off; every other column gets a boolean
    ``keep_<col>`` choice. The result preserves the canonical ``TABULAR_COLUMNS`` order.
    """
    keep = set(always_keep)
    for col in TABULAR_COLUMNS:
        if col in keep:
            continue
        if trial.suggest_categorical(f"keep_{col}", [True, False]):
            keep.add(col)
    return tuple(c for c in TABULAR_COLUMNS if c in keep)


def objective(
    trial,
    splits: dict[str, SplitData],
    classes: list[str],
    cfg: TuningConfig,
    scale_numeric: bool,
    use_embedding: bool,
    seed: int,
) -> float:
    """Validation macro-F1 for one trial. Reads only ``train`` and ``val`` (never ``test``)."""
    params = sample_params(trial, cfg.search_space)
    if cfg.select_features:
        selected = sample_selected_columns(trial, tuple(cfg.always_keep))
    else:
        selected = TABULAR_COLUMNS
    preprocessor = build_preprocessor(scale_numeric, selected_columns=selected).fit(
        splits["train"].tabular
    )
    spec = ModelSpec(type=cfg.target_model, params=params)
    estimator = fit_one(spec, splits, preprocessor, use_embedding, seed)
    val_metrics = evaluate_on(estimator, splits["val"], preprocessor, use_embedding, classes)
    return float(val_metrics["macro_f1"])


def measure_latency_ms(
    bundle: ModelBundle,
    tabular: pd.DataFrame,
    embedding: np.ndarray | None,
    repeats: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Measure single-row inference latency (what ``/predict`` does), in milliseconds.

    Samples ``repeats`` random rows and times one ``predict_frame`` call each, returning
    p50/p95. This is the serving-relevant signal for the optimisation electivo.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, len(tabular), size=repeats)
    times: list[float] = []
    for i in idxs:
        row = tabular.iloc[[int(i)]]
        emb = embedding[[int(i)]] if embedding is not None else None
        start = time.perf_counter()
        predict_frame(bundle, row, emb)
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


def _baseline_spec(train_cfg: TrainConfig, target_model: str) -> ModelSpec:
    """The default params for ``target_model`` from train.yaml — the 'before' of the comparison."""
    for spec in train_cfg.models:
        if spec.type == target_model:
            return spec
    return ModelSpec(type=target_model, params={})


def _fit_bundle(
    spec: ModelSpec,
    selected: tuple[str, ...],
    splits: dict[str, SplitData],
    classes: list[str],
    scale_numeric: bool,
    use_embedding: bool,
    seed: int,
    embedding_dim: int | None,
    version: str,
    dataset_hash: str,
    config_hash: str,
) -> ModelBundle:
    """Fit one configuration on train, evaluate on test once, and pack a bundle."""
    preprocessor = build_preprocessor(scale_numeric, selected_columns=selected).fit(
        splits["train"].tabular
    )
    estimator = fit_one(spec, splits, preprocessor, use_embedding, seed)
    test_metrics = evaluate_on(estimator, splits["test"], preprocessor, use_embedding, classes)
    return ModelBundle(
        model=estimator,
        preprocessor=preprocessor,
        classes=classes,
        tabular_columns=list(selected),
        embedding_dim=embedding_dim,
        model_type=spec.type,
        model_version=version,
        dataset_hash=dataset_hash,
        train_config_hash=config_hash,
        metrics=test_metrics,
    )


def run_tuning(train_cfg: TrainConfig, dataset_cfg: DatasetConfig) -> ModelBundle:
    """Run the Optuna search, log the before/after to MLflow, export the best bundle."""
    import mlflow
    import optuna
    from optuna.samplers import TPESampler

    from src.models.train import _dataset_hash, _flatten_metrics, _sha256_file

    cfg = train_cfg.tuning
    if not cfg.enabled:
        raise SystemExit("tuning.enabled is false in train.yaml; nothing to do.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    manifest, features = load_dataset(dataset_cfg)
    classes = sorted(manifest["label"].unique().tolist())
    splits = split_dataset(manifest, features, classes)

    use_embedding = train_cfg.features.use_embedding
    scale_numeric = train_cfg.features.scale_numeric
    embedding_dim = int(features.shape[1]) if use_embedding else None
    seed = train_cfg.seed

    dataset_hash = _dataset_hash(dataset_cfg)
    config_hash = _sha256_file(
        TRAIN_CONFIG_PATH if TRAIN_CONFIG_PATH.is_file() else Path("/dev/null")
    )

    # The search reads train+val only; a deterministic sampler makes the run reproducible.
    search_splits = {"train": splits["train"], "val": splits["val"]}
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(
        lambda t: objective(t, search_splits, classes, cfg, scale_numeric, use_embedding, seed),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout_s,
    )

    best_params = {k: v for k, v in study.best_params.items() if not k.startswith("keep_")}
    if cfg.select_features:
        best_selected = tuple(
            c
            for c in TABULAR_COLUMNS
            if c in set(cfg.always_keep) or study.best_params.get(f"keep_{c}", False)
        )
    else:
        best_selected = TABULAR_COLUMNS

    # Fair before/after: baseline = default params + all features; tuned = best of the search.
    baseline_bundle = _fit_bundle(
        _baseline_spec(train_cfg, cfg.target_model),
        TABULAR_COLUMNS,
        splits,
        classes,
        scale_numeric,
        use_embedding,
        seed,
        embedding_dim,
        f"opt-baseline-{config_hash[:8]}",
        dataset_hash,
        config_hash,
    )
    tuned_bundle = _fit_bundle(
        ModelSpec(type=cfg.target_model, params=best_params),
        best_selected,
        splits,
        classes,
        scale_numeric,
        use_embedding,
        seed,
        embedding_dim,
        f"v1-tuned-{cfg.target_model}-{config_hash[:8]}",
        dataset_hash,
        config_hash,
    )

    test_tab, test_emb = splits["test"].tabular, splits["test"].embedding
    baseline_lat = measure_latency_ms(baseline_bundle, test_tab, test_emb)
    tuned_lat = measure_latency_ms(tuned_bundle, test_tab, test_emb)

    mlflow.set_tracking_uri(train_cfg.mlflow.tracking_uri)
    mlflow.set_experiment("optimization-v1")
    is_registry_capable = not train_cfg.mlflow.tracking_uri.startswith("file")

    for name, bundle, latency in (
        ("baseline", baseline_bundle, baseline_lat),
        ("tuned-best", tuned_bundle, tuned_lat),
    ):
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model_type": cfg.target_model,
                    "seed": seed,
                    "use_embedding": use_embedding,
                    "n_features": len(bundle.tabular_columns),
                    "selected_columns": ",".join(bundle.tabular_columns),
                    **(bundle.model.get_params() if hasattr(bundle.model, "get_params") else {}),
                }
            )
            mlflow.set_tags({"dataset_hash": dataset_hash, "train_config_hash": config_hash})
            mlflow.log_metrics(_flatten_metrics(bundle.metrics, "test"))
            mlflow.log_metrics({f"latency_{k}": v for k, v in latency.items()})

    delta_f1 = tuned_bundle.metrics["macro_f1"] - baseline_bundle.metrics["macro_f1"]
    print(
        f"\nbaseline  test macro-F1={baseline_bundle.metrics['macro_f1']:.3f} "
        f"p50={baseline_lat['p50_ms']:.2f}ms p95={baseline_lat['p95_ms']:.2f}ms "
        f"(features={len(baseline_bundle.tabular_columns)})"
    )
    print(
        f"tuned     test macro-F1={tuned_bundle.metrics['macro_f1']:.3f} "
        f"p50={tuned_lat['p50_ms']:.2f}ms p95={tuned_lat['p95_ms']:.2f}ms "
        f"(features={len(tuned_bundle.tabular_columns)}: {', '.join(tuned_bundle.tabular_columns)})"
    )
    print(f"Δ macro-F1 = {delta_f1:+.3f}  |  best params: {best_params}")

    if delta_f1 < 0:
        print(
            "warning: tuned model did not beat the baseline on test; "
            "keeping the tuned bundle for traceability but review the search space."
        )

    if is_registry_capable:
        with mlflow.start_run(run_name="tuned-best-registered"):
            mlflow.sklearn.log_model(
                tuned_bundle.model, name="model", registered_model_name="soccernet-events-v1"
            )

    model_dir = train_cfg.paths.resolved("model_dir")
    bundle_path = save_bundle(tuned_bundle, model_dir)
    print(f"exported tuned bundle → {bundle_path}")
    return tuned_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune the classifier with Optuna (Phase 3).")
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG_PATH)
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG_PATH)
    args = parser.parse_args()

    train_cfg = TrainConfig.from_yaml(args.config)
    dataset_cfg = DatasetConfig.from_yaml(args.dataset_config)
    run_tuning(train_cfg, dataset_cfg)


if __name__ == "__main__":
    main()
