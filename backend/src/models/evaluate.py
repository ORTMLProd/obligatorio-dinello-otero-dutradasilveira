"""Evaluation metrics for the classifier (invariant 5 — handle imbalance explicitly).

``background`` dominates, so we never report bare accuracy: per-class precision/recall/F1,
macro-F1 and per-class PR-AUC (one-vs-rest) are the headline metrics. The small test split
can leave a class with no positives, so PR-AUC is computed defensively per class.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
) -> dict:
    """Compute per-class and aggregate metrics.

    Args:
        y_true, y_pred: integer-encoded labels (index into ``classes``).
        y_proba: probability matrix, columns ordered as ``classes``.
        classes: class names in column order.

    Returns:
        A JSON-serializable dict: ``per_class`` (precision/recall/f1/support/pr_auc),
        ``macro_f1`` and ``n``. No bare accuracy is exposed as a headline.
    """
    labels = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    y_true_bin = label_binarize(y_true, classes=labels)
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(classes):
        # PR-AUC is undefined for a class with no positive in y_true (small test split).
        pr_auc = (
            float(average_precision_score(y_true_bin[:, i], y_proba[:, i]))
            if y_true_bin[:, i].sum() > 0
            else None
        )
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "pr_auc": pr_auc,
        }

    return {
        "per_class": per_class,
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "n": int(len(y_true)),
    }


def save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_path: Path,
) -> Path | None:
    """Render the confusion matrix to ``out_path``.

    Plotting is best-effort: if matplotlib is unavailable (e.g. a slim env) the run still
    succeeds without the figure. Returns the path written, or ``None`` if skipped.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title("Matriz de confusión (test)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
