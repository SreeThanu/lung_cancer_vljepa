import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class SSLDataset(Dataset):
    """
    Unlabeled patch dataset for JEPA pretraining.
    """

    def __init__(self, patch_paths, augment=False):
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]

        data = np.load(patch_path)
        volume = torch.tensor(data["patch"], dtype=torch.float32)

        volume = volume.unsqueeze(0)  # [1, D, H, W]

        return volume
