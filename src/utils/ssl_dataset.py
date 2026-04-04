import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SSLDataset(Dataset):
    """Unlabeled patch dataset for JEPA pretraining."""

    def __init__(self, patch_paths, augment=False):
        self.patch_paths = patch_paths
        self.augment = augment

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]

        data = np.load(patch_path)
        volume = torch.tensor(data["patch"], dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]

        if self.augment:
            # Random flips across depth/height/width
            for dim in (1, 2, 3):
                if random.random() > 0.5:
                    volume = torch.flip(volume, dims=[dim])
        
            # Random 90-degree rotations in all three planes
            k = random.randint(0, 3)
            if k > 0:
                volume = torch.rot90(volume, k=k, dims=[2, 3])  # axial
        
            k2 = random.randint(0, 3)                           # ← add this
            if k2 > 0:                                          # ← add this
                volume = torch.rot90(volume, k=k2, dims=[1, 3]) # ← add this (sagittal)
        
            k3 = random.randint(0, 3)                           # ← add this
            if k3 > 0:                                          # ← add this
                volume = torch.rot90(volume, k=k3, dims=[1, 2]) # ← add this (coronal)
        
            # Gaussian noise
            if random.random() > 0.5:
                noise = torch.randn_like(volume) * 0.01
                volume = (volume + noise).clamp(0, 1)
        
            # Random intensity scaling
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                volume = (volume * scale).clamp(0, 1)

        return volume
