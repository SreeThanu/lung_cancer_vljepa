"""
LIDC-IDRI nodule dataset for DenseNet3D classification.

Protocol (published):
  - One sample per nodule row in labels_v2.csv. No per-patient selection.
  - Input patch size stored on disk: 40×40×40 (center-cropped from 96³).
  - Train: random-crop 40→32, independent random flip on each axis, 4³ voxel dropout.
  - Val  : center-crop 40→32, no augmentation.
  - Output: (1, 32, 32, 32) float32 tensor, asserted on every __getitem__.
  - RAM cache: when cache_in_ram=True, entire dataset is loaded into a single
    numpy array at init; __getitem__ indexes into it with zero disk I/O per epoch.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.paths import find_repo_root, get_data_dir

INPUT_SIZE  = 32   # model input after crop
PATCH_SIZE  = 40   # size on disk
_CROP_START = (PATCH_SIZE - INPUT_SIZE) // 2   # = 4
_CROP_END   = _CROP_START + INPUT_SIZE          # = 36


def _center_crop_32(arr: np.ndarray) -> np.ndarray:
    """Center-crop a (40,40,40) array to (32,32,32)."""
    return arr[_CROP_START:_CROP_END, _CROP_START:_CROP_END, _CROP_START:_CROP_END]


def _random_crop_32(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Random-crop a (40,40,40) array to (32,32,32).
    Offset drawn uniformly from [0, PATCH_SIZE - INPUT_SIZE] = [0, 8] per axis.
    Gives ±4-voxel (= ±4 mm at 1 mm/vox) translation jitter using real tissue.
    """
    margin = PATCH_SIZE - INPUT_SIZE   # = 8
    oz = int(rng.integers(0, margin + 1))
    oy = int(rng.integers(0, margin + 1))
    ox = int(rng.integers(0, margin + 1))
    return arr[oz:oz+INPUT_SIZE, oy:oy+INPUT_SIZE, ox:ox+INPUT_SIZE]


def _load_patch(path: Path) -> np.ndarray:
    """Load a 40³ .npz patch; returns float32 array."""
    arr = np.load(str(path))["patch"].astype(np.float32)
    if arr.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
        raise ValueError(
            f"{path}: expected shape ({PATCH_SIZE},{PATCH_SIZE},{PATCH_SIZE}), "
            f"got {arr.shape}"
        )
    return arr


class LIDCNoduleDataset(Dataset):
    """
    One sample per row in labels_v2.csv. No per-patient deduplication.

    Args:
        patch_dir:    Root of patches_40/ (absolute Path).
        df:           DataFrame slice (train or val rows).
        augment:      If True, apply train augmentation; else center-crop only.
        cache_in_ram: If True, preload all patches to RAM at init.
        seed:         RNG seed for augmentation reproducibility across workers.
    """

    def __init__(
        self,
        patch_dir: Path,
        df: pd.DataFrame,
        augment: bool = False,
        cache_in_ram: bool = True,
        seed: int = 42,
    ):
        self.patch_dir  = patch_dir
        self.augment    = augment
        self.seed       = seed

        self.df = df.reset_index(drop=True)

        # Build per-row paths and labels
        self.paths  = []
        self.labels = []
        missing = []

        for _, row in self.df.iterrows():
            p = patch_dir / str(row["patient_id"]) / f"{row['nodule_id']}.npz"
            if not p.exists():
                missing.append(str(p))
                continue
            self.paths.append(p)
            self.labels.append(int(row["label"]))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} patch(es) not found on disk. "
                f"First missing: {missing[0]}"
            )

        # Optionally preload entire dataset into RAM
        self._cache: Optional[np.ndarray] = None
        if cache_in_ram:
            print(f"  Preloading {len(self.paths)} patches to RAM...", end=" ", flush=True)
            self._cache = np.stack([_load_patch(p) for p in self.paths], axis=0)
            mb = self._cache.nbytes / 1024**2
            print(f"done ({mb:.1f} MB)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        # Per-worker deterministic RNG (avoids identical augmentation across workers)
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id if worker_info else 0
        rng = np.random.default_rng([self.seed, idx, worker_id])

        # Load patch (from cache or disk)
        if self._cache is not None:
            arr = self._cache[idx].copy()
        else:
            arr = _load_patch(self.paths[idx])

        # Spatial transform
        if self.augment:
            arr = _random_crop_32(arr, rng)
            # Independent random flip on each axis
            for ax in range(3):
                if rng.random() > 0.5:
                    arr = np.flip(arr, axis=ax).copy()
            # 4³ voxel dropout: zero a random 4-voxel block
            dz = int(rng.integers(0, INPUT_SIZE - 4 + 1))
            dy = int(rng.integers(0, INPUT_SIZE - 4 + 1))
            dx = int(rng.integers(0, INPUT_SIZE - 4 + 1))
            arr[dz:dz+4, dy:dy+4, dx:dx+4] = 0.0
        else:
            arr = _center_crop_32(arr)

        # Add channel dim → (1, 32, 32, 32)
        tensor = torch.from_numpy(arr[np.newaxis]).float()

        assert tensor.shape == (1, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE), (
            f"Dataset output shape {tensor.shape} != (1,{INPUT_SIZE},{INPUT_SIZE},{INPUT_SIZE})"
        )

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def build_dataloaders(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Build train and val DataLoaders for one CV fold.

    Args:
        cfg:      Full config dict (from densenet_config.yaml).
        train_df: Patient rows for this fold's train split.
        val_df:   Patient rows for this fold's val split.
    """
    repo     = find_repo_root()
    data_dir = get_data_dir(repo)
    patch_dir = data_dir / cfg["data"]["patch_dir"]

    cache = cfg["data"].get("cache_in_ram", True)
    seed  = cfg["project"].get("seed", 42)

    train_ds = LIDCNoduleDataset(patch_dir, train_df, augment=True,  cache_in_ram=cache, seed=seed)
    val_ds   = LIDCNoduleDataset(patch_dir, val_df,   augment=False, cache_in_ram=cache, seed=seed)

    bs          = cfg["data"]["batch_size"]
    nw          = cfg["data"].get("num_workers", 0)
    pin         = cfg["data"].get("pin_memory", False)
    persistent  = cfg["data"].get("persistent_workers", False) and nw > 0
    prefetch    = cfg["data"].get("prefetch_factor", 2) if nw > 0 else None

    loader_kwargs = dict(
        batch_size=bs,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persistent,
        **({"prefetch_factor": prefetch} if prefetch else {}),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader


# ── Test-time augmentation helpers ───────────────────────────────────────────

_TTA_OPS = {
    "identity": lambda x: x,
    "flip_x":   lambda x: torch.flip(x, dims=[-1]),
    "flip_y":   lambda x: torch.flip(x, dims=[-2]),
    "flip_z":   lambda x: torch.flip(x, dims=[-3]),
    "flip_xy":  lambda x: torch.flip(x, dims=[-1, -2]),
    "flip_xz":  lambda x: torch.flip(x, dims=[-1, -3]),
    "flip_yz":  lambda x: torch.flip(x, dims=[-2, -3]),
    "flip_xyz": lambda x: torch.flip(x, dims=[-1, -2, -3]),
}


def tta_predict(model, x: torch.Tensor, transforms: list, device: torch.device) -> torch.Tensor:
    """
    Apply TTA: average softmax probabilities across all requested transforms.

    Args:
        model:      Trained model in eval mode.
        x:          Input batch (B, 1, 32, 32, 32).
        transforms: List of transform names from tta_transforms config key.
        device:     Device.

    Returns:
        Averaged probabilities (B, 2).
    """
    probs_list = []
    model.eval()
    with torch.no_grad():
        for name in transforms:
            fn = _TTA_OPS[name]
            x_aug = fn(x.to(device))
            logits = model(x_aug)
            probs_list.append(torch.softmax(logits, dim=1))
    return torch.stack(probs_list, dim=0).mean(dim=0)
