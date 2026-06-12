"""Train the v0 baseline: late fusion [tabular ⊕ ResNet pooled] → class.

Pipeline (config-driven, seeded — invariant 6):
    load_dataset → split by manifest ``split`` column (per game_id, invariant 1)
    → fit preprocessor on TRAIN only (invariant 3)
    → for each model in train.yaml: fit, evaluate (val + test), log to MLflow
    → select best by validation macro-F1, export its bundle, register it.

The API never runs this; it loads the exported bundle. Model selection uses validation;
the test split is reported once for the winner (no selection on test).

Usage:
    uv run python -m src.models.train --config ../configs/train.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import DEFAULT_CONFIG_PATH as DATASET_CONFIG_PATH
from src.data.config import DatasetConfig
from src.data.dataset import load_dataset
from src.features.preprocess import assemble_matrix, build_preprocessor
from src.features.tabular import TABULAR_COLUMNS
from src.models.config import DEFAULT_CONFIG_PATH as TRAIN_CONFIG_PATH
from src.models.config import ModelSpec, TrainConfig
from src.models.evaluate import compute_metrics, save_confusion_matrix_png
from src.models.export import ModelBundle, save_bundle


@dataclass
class SplitData:
    """One split: tabular rows, row-aligned embedding, integer-encoded labels."""

    tabular: pd.DataFrame
    embedding: np.ndarray
    y: np.ndarray


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_hash(dataset_cfg: DatasetConfig) -> str:
    """Reuse the manifest content hash from the dataset summary for traceability."""
    summary_path = dataset_cfg.paths.resolved("summary_file")
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8")).get("content_sha256", "")
    return ""


def split_dataset(
    manifest: pd.DataFrame, features: np.ndarray, classes: list[str]
) -> dict[str, SplitData]:
    """Partition the row-aligned (manifest, features) into train/val/test SplitData."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    out: dict[str, SplitData] = {}
    for split in ("train", "val", "test"):
        rows = manifest[manifest["split"] == split]
        emb = features[rows["window_id"].to_numpy()]
        y = rows["label"].map(class_to_idx).to_numpy()
        out[split] = SplitData(
            tabular=rows[list(TABULAR_COLUMNS)].reset_index(drop=True), embedding=emb, y=y
        )
    return out


def build_estimator(spec: ModelSpec, seed: int):
    """Instantiate an estimator from its config spec."""
    if spec.type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, **spec.params)
    if spec.type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed, tree_method="hist", eval_metric="mlogloss", **spec.params
        )
    raise ValueError(f"unknown model type: {spec.type}")


def fit_one(
    spec: ModelSpec,
    splits: dict[str, SplitData],
    preprocessor,
    use_embedding: bool,
    seed: int,
):
    """Fit one estimator on the assembled train matrix; balance classes for trees."""
    from sklearn.utils.class_weight import compute_sample_weight

    train = splits["train"]
    emb = train.embedding if use_embedding else None
    x_train = assemble_matrix(train.tabular, emb, preprocessor)

    estimator = build_estimator(spec, seed)
    # LogReg balances via class_weight in its params; XGBoost has no class_weight,
    # so we pass balanced sample weights (invariant 5).
    if spec.type == "xgboost":
        estimator.fit(x_train, train.y, sample_weight=compute_sample_weight("balanced", train.y))
    else:
        estimator.fit(x_train, train.y)
    return estimator


def evaluate_on(
    estimator, split: SplitData, preprocessor, use_embedding: bool, classes: list[str]
) -> dict:
    """Compute metrics for a fitted estimator on one split."""
    emb = split.embedding if use_embedding else None
    matrix = assemble_matrix(split.tabular, emb, preprocessor)
    proba = np.asarray(estimator.predict_proba(matrix), dtype=np.float64)
    y_pred = proba.argmax(axis=1)
    return compute_metrics(split.y, y_pred, proba, classes)


def _flatten_metrics(metrics: dict, prefix: str) -> dict[str, float]:
    """Flatten the nested metrics dict into MLflow-friendly scalar keys."""
    flat: dict[str, float] = {f"{prefix}_macro_f1": metrics["macro_f1"]}
    for cls, m in metrics["per_class"].items():
        flat[f"{prefix}_f1_{cls}"] = m["f1"]
        flat[f"{prefix}_precision_{cls}"] = m["precision"]
        flat[f"{prefix}_recall_{cls}"] = m["recall"]
        if m["pr_auc"] is not None:
            flat[f"{prefix}_pr_auc_{cls}"] = m["pr_auc"]
    return flat


def run(train_cfg: TrainConfig, dataset_cfg: DatasetConfig) -> ModelBundle:
    """Train all configured models, log to MLflow, export and return the best bundle."""
    import mlflow

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    manifest, features = load_dataset(dataset_cfg)
    classes = sorted(manifest["label"].unique().tolist())
    splits = split_dataset(manifest, features, classes)

    use_embedding = train_cfg.features.use_embedding
    preprocessor = build_preprocessor(train_cfg.features.scale_numeric).fit(splits["train"].tabular)
    embedding_dim = int(features.shape[1]) if use_embedding else None

    dataset_hash = _dataset_hash(dataset_cfg)
    config_hash = _sha256_file(
        TRAIN_CONFIG_PATH if TRAIN_CONFIG_PATH.is_file() else Path("/dev/null")
    )

    mlflow.set_tracking_uri(train_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(train_cfg.mlflow.experiment_name)
    # The Model Registry needs a db/http backend; the deprecated file store cannot.
    is_registry_capable = not train_cfg.mlflow.tracking_uri.startswith("file")

    best: ModelBundle | None = None
    best_val_f1 = -1.0
    metrics_dir = train_cfg.paths.resolved("metrics_dir")

    for spec in train_cfg.models:
        estimator = fit_one(spec, splits, preprocessor, use_embedding, train_cfg.seed)
        val_metrics = evaluate_on(estimator, splits["val"], preprocessor, use_embedding, classes)
        test_metrics = evaluate_on(estimator, splits["test"], preprocessor, use_embedding, classes)

        print(
            f"\n[{spec.type}] val macro-F1={val_metrics['macro_f1']:.3f} "
            f"test macro-F1={test_metrics['macro_f1']:.3f}"
        )
        for cls, m in test_metrics["per_class"].items():
            print(
                f"  {cls:<13} P={m['precision']:.2f} R={m['recall']:.2f} "
                f"F1={m['f1']:.2f} (n={m['support']})"
            )

        with mlflow.start_run(run_name=spec.type):
            mlflow.log_params(
                {
                    "model_type": spec.type,
                    "seed": train_cfg.seed,
                    "use_embedding": use_embedding,
                    **spec.params,
                }
            )
            mlflow.set_tags({"dataset_hash": dataset_hash, "train_config_hash": config_hash})
            mlflow.log_metrics(_flatten_metrics(val_metrics, "val"))
            mlflow.log_metrics(_flatten_metrics(test_metrics, "test"))

            test_pred = _predict_labels(estimator, splits["test"], preprocessor, use_embedding)
            cm_path = save_confusion_matrix_png(
                splits["test"].y,
                test_pred,
                classes,
                metrics_dir / f"confusion_{spec.type}.png",
            )
            if cm_path is not None:
                mlflow.log_artifact(str(cm_path))

            reg_name = "soccernet-events-v0" if is_registry_capable else None
            mlflow.sklearn.log_model(estimator, name="model", registered_model_name=reg_name)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best = ModelBundle(
                model=estimator,
                preprocessor=preprocessor,
                classes=classes,
                tabular_columns=list(TABULAR_COLUMNS),
                embedding_dim=embedding_dim,
                model_type=spec.type,
                model_version=f"v0-{spec.type}-{config_hash[:8]}",
                dataset_hash=dataset_hash,
                train_config_hash=config_hash,
                metrics=test_metrics,
            )

    assert best is not None
    model_dir = train_cfg.paths.resolved("model_dir")
    bundle_path = save_bundle(best, model_dir)
    (metrics_dir).mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics_v0.json").write_text(
        json.dumps(best.metrics, indent=2), encoding="utf-8"
    )
    print(f"\nbest model: {best.model_type} (val macro-F1={best_val_f1:.3f}) → {bundle_path}")
    return best


def _predict_labels(estimator, split: SplitData, preprocessor, use_embedding: bool) -> np.ndarray:
    emb = split.embedding if use_embedding else None
    matrix = assemble_matrix(split.tabular, emb, preprocessor)
    return np.asarray(estimator.predict_proba(matrix)).argmax(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the v0 baseline classifier.")
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG_PATH)
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG_PATH)
    args = parser.parse_args()

    train_cfg = TrainConfig.from_yaml(args.config)
    dataset_cfg = DatasetConfig.from_yaml(args.dataset_config)
    run(train_cfg, dataset_cfg)


if __name__ == "__main__":
    main()
