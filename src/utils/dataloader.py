"""
Patch-based LIDC Dataset
- Uses SeriesUID (not patient_id)
- Compatible with 5-fold cross-validation
- Patch-level expansion
- Class imbalance handling
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from monai.transforms import Compose, RandRotate90, RandFlip


# ============================================================
# PATCH DATASET (SERIESUID BASED)
# ============================================================

class LIDCPatchDataset(Dataset):

    def __init__(
        self,
        processed_dir: str,
        metadata: pd.DataFrame,
        augment: bool = False,
    ):
        """
        Args:
            processed_dir: path to data/processed
            metadata: dataframe containing series_uid + label
            augment: whether to apply data augmentation
        """

        self.processed_dir = Path(processed_dir)
        self.metadata = metadata.reset_index(drop=True)
        self.augment = augment

        if augment:
            self.transforms = Compose([
                RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                RandFlip(prob=0.5, spatial_axis=2),
            ])
        else:
            self.transforms = None

        # Expand series-level metadata to patch-level samples
        self.samples = []

        for _, row in self.metadata.iterrows():

            series_uid = row["series_uid"]
            label = int(row["label"])

            patch_dir = self.processed_dir / "patches" / series_uid

            if not patch_dir.exists():
                continue

            for patch_file in patch_dir.glob("*.npz"):
                self.samples.append((patch_file, label))

        print(f"Loaded {len(self.samples)} patch samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        patch_path, label = self.samples[idx]

        data = np.load(patch_path)
        volume = torch.tensor(data["context"], dtype=torch.float32)

        volume = volume.unsqueeze(0)  # [1, D, H, W]

        if self.transforms:
            volume = self.transforms(volume)

        return volume, torch.tensor(label, dtype=torch.long)


# ============================================================
# CREATE CROSS-VALIDATION DATALOADERS
# ============================================================

def create_cv_dataloaders(config, fold_index: int):
    """
    Creates train/val loaders for specific CV fold.
    """

    

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]

    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    seed = config["project"]["seed"]
    n_splits = config["data"]["cross_validation_folds"]

    labels_path = Path(processed_dir) / "metadata" / "labels.csv"
    labels_df = pd.read_csv(labels_path)

    # ============================================================
    # STRATIFIED K-FOLD SPLIT (SERIES LEVEL)
    # ============================================================

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    splits = list(skf.split(labels_df["series_uid"], labels_df["label"]))

    train_idx, val_idx = splits[fold_index]

    train_df = labels_df.iloc[train_idx]
    val_df = labels_df.iloc[val_idx]

    # ============================================================
    # DATASETS
    # ============================================================

    train_dataset = LIDCPatchDataset(
        processed_dir,
        train_df,
        augment=True
    )

    val_dataset = LIDCPatchDataset(
        processed_dir,
        val_df,
        augment=False
    )

    # ============================================================
    # CLASS IMBALANCE HANDLING (PATCH-LEVEL)
    # ============================================================

    if config["finetuning"]["use_class_weights"]:

        labels = [label for _, label in train_dataset.samples]

        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts

        sample_weights = [class_weights[label] for label in labels]

        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        shuffle = False

    else:
        sampler = None
        shuffle = True

    # ============================================================
    # LOADERS
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
