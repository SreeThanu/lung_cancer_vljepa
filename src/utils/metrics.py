"""Medical imaging metrics for lung cancer classification."""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# CORE METRICS
# ============================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """Compute clinically relevant binary classification metrics."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    if y_proba is not None:
        y_proba = np.asarray(y_proba).astype(float)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold_from_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_sensitivity: float = None,
) -> Dict[str, float]:
    """
    Tune threshold from ROC.

    Strategy (default): maximize Youden's J (sensitivity + specificity - 1).
    If min_sensitivity is set: prefer thresholds with sensitivity >= that value,
    maximizing specificity among them; fall back to Youden's J if none qualify.
    """

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    y_proba = np.nan_to_num(y_proba, nan=0.5, posinf=1.0, neginf=0.0)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificity = 1.0 - fpr
    youden = tpr + specificity - 1.0

    if min_sensitivity is not None:
        valid = np.where(tpr >= min_sensitivity)[0]
        if valid.size > 0:
            best_idx = valid[np.argmax(specificity[valid])]
        else:
            best_idx = int(np.argmax(youden))
    else:
        best_idx = int(np.argmax(youden))

    threshold = float(thresholds[best_idx])

    return {
        "threshold": threshold,
        "sensitivity": float(tpr[best_idx]),
        "specificity": float(specificity[best_idx]),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def predict_with_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Convert probabilities to binary predictions using a custom threshold."""
    y_proba = np.asarray(y_proba).astype(float)
    return (y_proba >= threshold).astype(int)


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
):
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

    thresh = cm.max() / 2.0 if cm.size else 0.0
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
