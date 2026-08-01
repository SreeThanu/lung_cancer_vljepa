"""
10-fold patient-grouped cross-validation for DenseNet121-3D.

Key invariants enforced here:
  - Splits are by patient_id (GroupKFold), never by nodule. No patient appears
    in both train and val of the same fold.
  - Zero-leakage assertion: after splitting, both sets of patient IDs are
    checked to be disjoint. Script aborts if any overlap is found.
  - All metrics are computed per fold from argmax predictions AND from
    raw probabilities (for AUC).
  - TTA is applied at val time using tta_predict() from the dataloader module.
  - Fold models are saved as checkpoints/densenet/fold_{k}.pth.
  - Ensemble predictions are the average of per-fold softmax probabilities
    over all val rows (each nodule predicted exactly once, in its held-out fold).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.models.densenet_3d import build_densenet
from src.utils.dataloader import LIDCNoduleDataset, build_dataloaders, tta_predict
from src.utils.device import get_device
from src.utils.paths import find_repo_root, get_data_dir


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC, accuracy, sensitivity (recall), specificity, precision, F1.

    y_true: integer labels (0/1), shape (N,)
    y_prob: class-1 probability, shape (N,)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc":         float(roc_auc_score(y_true, y_prob)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(spec),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ── Warmup + step LR scheduler ───────────────────────────────────────────────

class WarmupStepLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, step_size, gamma):
        self.warmup = warmup_epochs
        self.step   = step_size
        self.gamma  = gamma

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            steps_done = (epoch - warmup_epochs) // step_size
            return gamma ** steps_done

        super().__init__(optimizer, lr_lambda)


# ── Single-fold trainer ───────────────────────────────────────────────────────

def train_one_fold(
    cfg: dict,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    save_dir: Path,
) -> Dict:
    """
    Train one CV fold. Returns dict with val metrics and per-row probabilities.
    """
    train_loader, val_loader = build_dataloaders(cfg, train_df, val_df)

    model = build_densenet(cfg)
    model = model.to(device)
    if cfg["training"].get("channels_last_3d", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last_3d)

    train_cfg = cfg["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = WarmupStepLR(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        step_size=train_cfg.get("lr_step_size", 30),
        gamma=train_cfg.get("lr_gamma", 0.5),
    )

    criterion = nn.CrossEntropyLoss()
    use_amp   = train_cfg.get("amp", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16
    scaler    = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    epochs             = train_cfg["epochs"]
    patience           = train_cfg.get("early_stopping_patience", 40)
    best_val_auc       = -1.0
    epochs_no_improve  = 0
    best_ckpt_path     = save_dir / f"fold_{fold}.pth"

    tta_transforms = cfg.get("tta_transforms", ["identity"]) if cfg.get("tta", False) else ["identity"]

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            if cfg["training"].get("channels_last_3d", False) and device.type == "cuda":
                x = x.to(device=device, memory_format=torch.channels_last_3d)
            else:
                x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss   = criterion(logits, y)
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # ── Val (with TTA) ─────────────────────────────────────
        model.eval()
        all_probs  = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                probs = tta_predict(model, x, tta_transforms, device)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())

        y_prob_all = np.concatenate(all_probs, axis=0)[:, 1]
        y_true_all = np.concatenate(all_labels, axis=0)
        val_auc    = float(roc_auc_score(y_true_all, y_prob_all))

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"  epoch {epoch+1:03d}/{epochs}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}", flush=True)

        # ── Checkpoint best ────────────────────────────────────
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1} (best AUC={best_val_auc:.4f})")
            break

    # ── Final eval from best checkpoint ───────────────────────
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            probs = tta_predict(model, x, tta_transforms, device)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_prob_all = np.concatenate(all_probs, axis=0)[:, 1]
    y_true_all = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(y_true_all, y_prob_all)
    metrics["best_epoch"] = int(ckpt["epoch"])

    print(f"  [fold {fold}] AUC={metrics['auc']:.4f}  acc={metrics['accuracy']:.4f}  "
          f"sens={metrics['sensitivity']:.4f}  spec={metrics['specificity']:.4f}  "
          f"f1={metrics['f1']:.4f}")

    return {
        "metrics": metrics,
        "y_prob":  y_prob_all,
        "y_true":  y_true_all,
    }


# ── Main CV runner ────────────────────────────────────────────────────────────

def run_cross_validation(cfg: dict) -> Dict:
    """
    10-fold patient-grouped CV.

    Returns dict with per-fold metrics, ensemble metrics, and paths to fold checkpoints.
    """
    repo     = find_repo_root()
    data_dir = get_data_dir(repo)
    labels_path = data_dir / cfg["data"]["labels_csv"]

    df = pd.read_csv(labels_path)
    required_cols = {"patient_id", "nodule_id", "label"}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"labels_v2.csv missing columns: {missing_cols}")

    device = get_device()

    n_folds  = cfg.get("cv_folds", 10)
    save_dir = repo / cfg["training"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    groups = df["patient_id"].values
    gkf    = GroupKFold(n_splits=n_folds)

    fold_results: List[Dict] = []
    # For ensemble: accumulate predictions in original df order
    ensemble_probs  = np.zeros(len(df))
    ensemble_filled = np.zeros(len(df), dtype=bool)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["label"], groups), start=1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_folds}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx].copy()
        val_df   = df.iloc[val_idx].copy()

        # ── Zero-leakage assertion ─────────────────────────────
        train_patients = set(train_df["patient_id"].unique())
        val_patients   = set(val_df["patient_id"].unique())
        overlap = train_patients & val_patients
        assert len(overlap) == 0, (
            f"LEAKAGE DETECTED in fold {fold}: "
            f"{len(overlap)} patient(s) in both train and val — {list(overlap)[:5]}"
        )
        print(f"  train: {len(train_df)} nodules / {len(train_patients)} patients")
        print(f"  val  : {len(val_df)} nodules  / {len(val_patients)} patients")

        result = train_one_fold(cfg, fold, train_df, val_df, device, save_dir)
        fold_results.append(result)

        # Store val predictions in ensemble accumulator (original df index)
        ensemble_probs[val_idx]  = result["y_prob"]
        ensemble_filled[val_idx] = True

    # ── Summary ────────────────────────────────────────────────
    metric_keys = ["auc", "accuracy", "sensitivity", "specificity", "precision", "f1"]
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for key in metric_keys:
        vals = [r["metrics"][key] for r in fold_results]
        print(f"  {key:<12} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ── Ensemble metrics ───────────────────────────────────────
    assert ensemble_filled.all(), "Not all nodules received a prediction — check fold coverage."
    y_true_all = df["label"].values
    ensemble_metrics = compute_metrics(y_true_all, ensemble_probs)
    print(f"\nEnsemble (fold-averaged, each nodule predicted once in its val fold):")
    for key in metric_keys:
        print(f"  {key:<12} {ensemble_metrics[key]:.4f}")
    print(f"{'='*60}")

    # ── Save fold summary CSV ──────────────────────────────────
    rows = []
    for i, r in enumerate(fold_results, start=1):
        row = {"fold": i}
        row.update(r["metrics"])
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = save_dir / "cv_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nFold summary saved → {summary_path}")

    return {
        "fold_results":      fold_results,
        "ensemble_metrics":  ensemble_metrics,
        "ensemble_probs":    ensemble_probs,
        "fold_summary_path": summary_path,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    repo     = find_repo_root()
    cfg_path = repo / "configs" / "densenet_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_cross_validation(cfg)


if __name__ == "__main__":
    main()
