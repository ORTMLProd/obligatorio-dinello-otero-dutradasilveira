"""Train the clip classifier (Fase 3.5): frozen ResNet18 + head, head-only training.

``fit``/``evaluate_clips`` are the testable core (work with any nn.Module). ``run`` is the
orchestration (real model, dataset, MLflow, export) and is verified live, like train.py's run().
Trains once without augmentation and once with it, logs both to MLflow and reports the delta,
then exports the best (val macro-F1) bundle.

Usage:
    uv run python -m src.models.train_clips --config ../configs/train_clips.yaml
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.clips_dataset import ClipsDataset
from src.models.clip_config import DEFAULT_CLIP_CONFIG_PATH, ClipTrainConfig
from src.models.clip_export import ClipModelMeta, save_clip_bundle
from src.models.clip_model import build_clip_model, build_transforms, pick_device
from src.models.evaluate import compute_metrics, save_confusion_matrix_png


def evaluate_clips(
    model: nn.Module, loader: DataLoader, classes: list[str], device: torch.device
) -> dict:
    """Run the model over a loader and compute per-class metrics (invariant 5)."""
    model.eval()
    model.to(device)
    ys: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    with torch.no_grad():
        for clips, labels in loader:
            logits = model(clips.to(device))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
            ys.append(labels.numpy())
    y_true = np.concatenate(ys)
    proba = np.concatenate(probas).astype(np.float64)
    return compute_metrics(y_true, proba.argmax(axis=1), proba, classes)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    classes: list[str],
    epochs: int,
    patience: int,
    lr: float,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, float]:
    """Train the head with early stopping on validation macro-F1. Returns (best_model, best_f1)."""
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_f1, best_state, since = -1.0, copy.deepcopy(model.state_dict()), 0
    for _epoch in range(epochs):
        model.train()
        for clips, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(clips.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        val_f1 = evaluate_clips(model, val_loader, classes, device)["macro_f1"]
        if val_f1 > best_f1:
            best_f1, best_state, since = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_f1


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(labels: np.ndarray, classes: list[str]) -> torch.Tensor:
    from sklearn.utils.class_weight import compute_class_weight

    present = np.array([classes.index(label_str) for label_str in labels])
    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=present)
    return torch.tensor(weights, dtype=torch.float32)


def _make_loader(
    manifest: pd.DataFrame,
    cfg: ClipTrainConfig,
    classes: list[str],
    augment: bool,
    processed_dir: Path,
    shuffle: bool,
) -> DataLoader:
    transform = build_transforms(augment, cfg.frame_size, cfg.normalize.mean, cfg.normalize.std)
    ds = ClipsDataset(manifest, processed_dir, classes, transform)
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle)


def _predict_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for clips, _labels in loader:
            preds.append(torch.softmax(model(clips.to(device)), dim=1).cpu().numpy().argmax(axis=1))
    return np.concatenate(preds)


def _loader_labels(loader: DataLoader) -> np.ndarray:
    return np.concatenate([labels.numpy() for _clips, labels in loader])


def run(cfg: ClipTrainConfig) -> ClipModelMeta:
    """Train no-aug and aug runs, log to MLflow, export the best bundle. Verified live."""
    import mlflow
    import mlflow.pytorch

    _seed_everything(cfg.seed)
    device = pick_device()
    processed_dir = cfg.paths.resolved("processed_dir")
    manifest = pd.read_parquet(cfg.paths.resolved("manifest"))
    classes = sorted(manifest["label"].unique().tolist())

    splits = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    weights = _class_weights(splits["train"]["label"].to_numpy(), classes)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    # The MLflow Model Registry needs a DB-backed store (sqlite/postgres); a plain file
    # store can't register. Guard registration the same way v0 does (see train.py).
    is_registry_capable = not cfg.mlflow.tracking_uri.startswith("file")
    metrics_dir = cfg.paths.resolved("metrics_dir")

    best_meta: ClipModelMeta | None = None
    best_model: nn.Module | None = None
    best_val = -1.0
    results: dict[str, float] = {}

    for augment in (False, True):
        name = "clips-aug" if augment else "clips-noaug"
        train_loader = _make_loader(splits["train"], cfg, classes, augment, processed_dir, True)
        val_loader = _make_loader(splits["val"], cfg, classes, False, processed_dir, False)
        test_loader = _make_loader(splits["test"], cfg, classes, False, processed_dir, False)

        model = build_clip_model(
            len(classes), cfg.head.hidden, cfg.head.dropout, cfg.pooling, cfg.backbone
        )
        model, val_f1 = fit(
            model,
            train_loader,
            val_loader,
            classes,
            cfg.train.epochs,
            cfg.train.patience,
            cfg.train.lr,
            weights,
            device,
        )
        test_metrics = evaluate_clips(model, test_loader, classes, device)
        results[name] = test_metrics["macro_f1"]

        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "backbone": cfg.backbone,
                    "pooling": cfg.pooling,
                    "k": cfg.k,
                    "augment": augment,
                    "lr": cfg.train.lr,
                    "epochs": cfg.train.epochs,
                    "batch_size": cfg.train.batch_size,
                    "seed": cfg.seed,
                }
            )
            flat = {f"test_f1_{c}": m["f1"] for c, m in test_metrics["per_class"].items()}
            flat["test_macro_f1"] = test_metrics["macro_f1"]
            flat["val_macro_f1"] = val_f1
            mlflow.log_metrics(flat)
            test_pred = _predict_labels(model, test_loader, device)
            cm = save_confusion_matrix_png(
                _loader_labels(test_loader),
                test_pred,
                classes,
                metrics_dir / f"confusion_{name}.png",
            )
            if cm is not None:
                mlflow.log_artifact(str(cm))

            # Version the image model in the MLflow Model Registry (elective: ML traceability).
            # Logs the full nn.Module (frozen backbone + trained head); both the no-aug and aug
            # runs become versions of the same registered model, so the elective is validated on
            # the image model alone — no dependency on v0.
            reg_name = "soccernet-events-clips-v1" if is_registry_capable else None
            mlflow.pytorch.log_model(model, name="model", registered_model_name=reg_name)

        if val_f1 > best_val:
            best_val = val_f1
            best_model = model
            best_meta = ClipModelMeta(
                backbone=cfg.backbone,
                pooling=cfg.pooling,
                classes=classes,
                k=cfg.k,
                frame_size=cfg.frame_size,
                hidden=cfg.head.hidden,
                dropout=cfg.head.dropout,
                normalize_mean=cfg.normalize.mean,
                normalize_std=cfg.normalize.std,
                model_version=f"clips-v1-{name}",
                metrics=test_metrics,
            )

    assert best_meta is not None and best_model is not None
    save_clip_bundle(best_model, best_meta, cfg.paths.resolved("model_dir"))
    delta = results.get("clips-aug", 0.0) - results.get("clips-noaug", 0.0)
    print(f"\nno-aug test macro-F1={results.get('clips-noaug'):.3f}")
    print(f"aug    test macro-F1={results.get('clips-aug'):.3f}  (Δ augmentation = {delta:+.3f})")
    print(f"exported best bundle (val macro-F1={best_val:.3f}) → {cfg.paths.resolved('model_dir')}")
    return best_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-frame clip classifier.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CLIP_CONFIG_PATH)
    args = parser.parse_args()
    run(ClipTrainConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
