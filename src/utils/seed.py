"""
Global Reproducibility Utilities

Ensures deterministic behavior for:
- PyTorch
- CUDA
- NumPy
- Python random
- Multi-worker DataLoader

Use this in main training script:
    from utils.seed import set_global_seed
    set_global_seed(42, deterministic=True)
"""

import os
import random
import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Set seed for full experiment reproducibility.

    Args:
        seed (int): Random seed
        deterministic (bool): Enforce deterministic behavior
    """

    print(f"Setting global seed: {seed}")

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    if deterministic:
        print("Enabling deterministic mode (may reduce performance)")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # For full reproducibility (CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Ensure hash-based operations deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)

    print("Reproducibility configured successfully.")


def seed_worker(worker_id):
    """
    Ensures reproducibility in DataLoader workers.
    Use inside DataLoader:

    DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
