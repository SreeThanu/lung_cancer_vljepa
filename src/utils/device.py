"""
Device resolver for the lung_cancer_vljepa project.

Priority: cuda → mps → cpu.
Prints the selected device loudly so it is visible in notebook/script output.
"""

import torch


def get_device() -> torch.device:
    """
    Resolve and return the best available device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.
    Prints a clear one-liner so there is no silent fallback.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[device] CUDA selected — {name} ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] MPS selected — Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("[device] CPU selected — no GPU available")

    return device
