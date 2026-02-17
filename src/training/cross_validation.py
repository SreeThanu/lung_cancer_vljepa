"""
5-Fold Cross Validation for Lung Cancer Classification

Features:
- Stratified K-Fold (patient-level safe)
- Fresh model per fold
- Fold-wise checkpoint saving
- Mean ± Std ROC-AUC reporting
- Publication-ready statistical summary
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from utils.seed import set_global_seed
from utils.config import Config
from utils.metrics import compute_classification_metrics
from utils.dataloader import LIDCDataset
from training.evaluate import ClassificationTrainer


# ============================================================
# CROSS VALIDATION CLASS
# ============================================================

class CrossValidator:

    def __init__(self, config: Config, metadata_path: str, n_splits: int = 5):
        self.config = config
        self.metadata_path = metadata_path
        self.n_splits = n_splits

        self.device = torch.device(
            config.get("project.device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        self.results = []

        # Ensure reproducibility
        if config.get("project.deterministic", False):
            set_global_seed(config.get("project.seed", 42))

    # --------------------------------------------------------

    def run(self):

        print(f"\nRunning {self.n_splits}-Fold Cross Validation\n")

        metadata = pd.read_csv(self.metadata_path)

        if "label" not in metadata.columns:
            raise ValueError("Metadata CSV must contain 'label' column.")

        labels = metadata["label"].values

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.config.get("project.seed", 42),
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(metadata, labels)):

            print("=" * 60)
            print(f"FOLD {fold + 1}/{self.n_splits}")
            print("=" * 60)

            # --------------------------------------------------
            # Dataset split
            # --------------------------------------------------

            train_dataset = LIDCDataset(
                processed_dir=self.config.get("data.processed_dir"),
                metadata_path=self.metadata_path,
                split=None,
                augment=True,
            )

            val_dataset = LIDCDataset(
                processed_dir=self.config.get("data.processed_dir"),
                metadata_path=self.metadata_path,
                split=None,
                augment=False,
            )

            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(val_dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=self.config.get("data.batch_size", 4),
                shuffle=True,
                num_workers=self.config.get("data.num_workers", 4),
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.config.get("data.batch_size", 4),
                shuffle=False,
                num_workers=self.config.get("data.num_workers", 4),
                pin_memory=True,
            )

            # --------------------------------------------------
            # Fresh model per fold
            # --------------------------------------------------

            trainer = ClassificationTrainer(self.config)

            trainer.train(
                train_loader,
                val_loader,
                epochs=self.config.get("finetuning.epochs", 50),
            )

            # --------------------------------------------------
            # Evaluation
            # --------------------------------------------------

            val_metrics, _, _, _ = trainer.evaluate(val_loader)

            fold_auc = val_metrics.get("roc_auc", 0.0)
            self.results.append(fold_auc)

            print(f"\nFold {fold + 1} ROC-AUC: {fold_auc:.4f}")

            # Save fold checkpoint
            save_dir = Path(self.config.get("pretraining.save_dir", "./checkpoints"))
            save_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model": trainer.model.state_dict(),
                    "fold": fold + 1,
                    "roc_auc": fold_auc,
                },
                save_dir / f"cv_fold_{fold + 1}.pth",
            )

        # ------------------------------------------------------
        # Final CV Summary
        # ------------------------------------------------------

        mean_auc = np.mean(self.results)
        std_auc = np.std(self.results)

        print("\n" + "=" * 60)
        print("CROSS VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Mean ROC-AUC : {mean_auc:.4f}")
        print(f"Std ROC-AUC  : {std_auc:.4f}")
        print("=" * 60)

        return mean_auc, std_auc


# ============================================================
# MAIN
# ============================================================

def main():

    config = Config("configs/jepa_config.yaml")

    metadata_path = (
        Path(config.get("data.processed_dir")) / "metadata" / "labels.csv"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")

    cross_validator = CrossValidator(
        config=config,
        metadata_path=str(metadata_path),
        n_splits=5,
    )

    cross_validator.run()


if __name__ == "__main__":
    main()
