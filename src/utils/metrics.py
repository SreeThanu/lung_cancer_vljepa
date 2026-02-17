"""
Medical imaging metrics for lung cancer classification.
Research-grade evaluation metrics.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    average_precision_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
from typing import Tuple, Dict


# ============================================================
# CORE METRICS
# ============================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute comprehensive medical classification metrics.
    """

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics["f1"] = f1_score(
        y_true, y_pred, average="binary", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Edge case: only one class present
        tn = fp = fn = tp = 0

    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # MCC (important for imbalanced medical datasets)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # ROC-AUC and PR-AUC
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except:
            metrics["pr_auc"] = 0.0

    return metrics


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot ROC curve.
    """

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fpr, tpr, auc


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
    figsize: Tuple[int, int] = (6, 5),
):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        class_names = ["Benign", "Malignant"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_metrics_summary(metrics: Dict[str, float]):
    """
    Pretty formatted metric summary for publication.
    """

    print("\n" + "=" * 60)
    print("LUNG CANCER CLASSIFICATION RESULTS")
    print("=" * 60)

    ordered_keys = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "mcc",
        "ppv",
        "npv",
    ]

    for key in ordered_keys:
        if key in metrics:
            print(f"{key.upper():20s}: {metrics[key]:.4f}")

    print("=" * 60 + "\n")
