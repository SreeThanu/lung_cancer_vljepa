"""
Patch-based LIDC datasets for downstream classification.

Key behaviors:
- Patch directories on disk are named by patient_id (e.g. LIDC-IDRI-0157),
  NOT by series_uid. Both dataset classes use row["patient_id"] for path lookups.
- Supports series-level split to avoid leakage
- Loads saved nodule-centered patches from .npz
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# ============================================================
# PATCH DATASET (PATCH-LEVEL EXPANSION)
# ============================================================

class LIDCPatchDataset(Dataset):
    """Patch-level dataset expanded from series-level metadata.

    Each row in metadata must have:
        - patient_id  : e.g. "LIDC-IDRI-0157"  (used to locate patch dir)
        - series_uid  : kept for reference / grouping
        - label       : 0 or 1
    """

    def __init__(
        self,
        processed_dir: str,
        metadata: pd.DataFrame,
        augment: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.metadata = metadata.reset_index(drop=True)
        self.augment = augment

        self.samples = []  # list of (patch_file, label, series_uid)

        for _, row in self.metadata.iterrows():
            series_uid = str(row["series_uid"])
            patient_id = str(row["patient_id"])          # ← KEY FIX
            label = int(row["label"])

            patch_dir = self.processed_dir / "patches" / patient_id  # ← KEY FIX
            if not patch_dir.exists():
                continue

            for patch_file in sorted(patch_dir.glob("*.npz")):
                self.samples.append((patch_file, label, series_uid))

        print(f"Loaded {len(self.samples)} patch samples from {len(self.metadata)} series rows.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_patch_array(npz_file: Path) -> np.ndarray:
        data = np.load(npz_file)
        if "patch" in data:
            return data["patch"]
        elif "context" in data:
            return data["context"]
        raise KeyError(
            f"Expected 'patch' or 'context' key in {npz_file}, found: {list(data.keys())}"
        )

    @staticmethod
    def _augment(volume: torch.Tensor) -> torch.Tensor:
        # Random flips across all 3 spatial axes
        for dim in (1, 2, 3):
            if random.random() > 0.5:
                volume = torch.flip(volume, dims=[dim])

        # Random 90-degree rotation in axial plane
        k = random.randint(0, 3)
        if k > 0:
            volume = torch.rot90(volume, k=k, dims=[2, 3])

        # Light Gaussian noise
        if random.random() > 0.5:
            volume = (volume + torch.randn_like(volume) * 0.01).clamp(0, 1)

        return volume

    def __getitem__(self, idx: int):
        patch_path, label, series_uid = self.samples[idx]

        arr = self._load_patch_array(patch_path)
        volume = torch.tensor(arr, dtype=torch.float32)

        if volume.ndim == 3:
            volume = volume.unsqueeze(0)  # add channel dim → (1, D, H, W)

        if self.augment:
            volume = self._augment(volume)

        return volume, torch.tensor(label, dtype=torch.long), series_uid


# ============================================================
# SERIES-LEVEL DATASET (ONE RANDOM PATCH / SERIES / EPOCH)
# ============================================================

class LIDCClassificationDataset(Dataset):
    """Series-level dataset that samples one random patch per series on each access.

    Each row in metadata must have:
        - patient_id  : e.g. "LIDC-IDRI-0157"  (used to locate patch dir)
        - series_uid  : kept for reference / grouping
        - label       : 0 or 1
    """

    def __init__(self, processed_dir: str, metadata: pd.DataFrame, augment: bool = False):
        self.processed_dir = Path(processed_dir)
        self.augment = augment

        rows = []
        missing = 0
        for _, row in metadata.reset_index(drop=True).iterrows():
            series_uid = str(row["series_uid"])
            patient_id = str(row["patient_id"])          # ← KEY FIX
            label = int(row["label"])

            patch_dir = self.processed_dir / "patches" / patient_id  # ← KEY FIX
            if not patch_dir.exists():
                missing += 1
                continue

            patch_files = sorted(patch_dir.glob("*.npz"))
            if patch_files:
                rows.append((series_uid, patient_id, label, patch_files))

        self.rows = rows
        print(
            f"LIDCClassificationDataset: {len(self.rows)} series loaded "
            f"({missing} patch dirs not found on disk)."
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        series_uid, patient_id, label, patch_files = self.rows[idx]
        patch_file = random.choice(patch_files)

        arr = LIDCPatchDataset._load_patch_array(patch_file)
        volume = torch.tensor(arr, dtype=torch.float32)

        if volume.ndim == 3:
            volume = volume.unsqueeze(0)

        if self.augment:
            volume = LIDCPatchDataset._augment(volume)

        return volume, torch.tensor(label, dtype=torch.long), series_uid


# ============================================================
# CV DATALOADER FACTORY
# ============================================================

def create_cv_dataloaders(config, fold_index: int, num_workers_override: Optional[int] = None):
    """Create train/val loaders for a specific CV fold (series-level split)."""

    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / config["data"]["processed_dir"]

    batch_size = config["data"]["batch_size"]
    num_workers = (
        config["data"]["num_workers"] if num_workers_override is None else num_workers_override
    )
    seed = config["project"]["seed"]
    n_splits = config["data"]["cross_validation_folds"]

    labels_path = Path(processed_dir) / "metadata" / "labels.csv"
    labels_df = pd.read_csv(labels_path)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(labels_df["series_uid"], labels_df["label"]))

    train_idx, val_idx = splits[fold_index]
    train_df = labels_df.iloc[train_idx]
    val_df = labels_df.iloc[val_idx]

    train_dataset = LIDCPatchDataset(processed_dir, train_df, augment=True)
    val_dataset = LIDCPatchDataset(processed_dir, val_df, augment=False)

    if config["finetuning"].get("use_class_weights", False):
        labels = [label for _, label, _ in train_dataset.samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

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