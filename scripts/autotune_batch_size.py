"""
Phase 4 — Auto-tune batch size for the lab GPU.

Binary-searches the largest batch size that fits in VRAM with a 20% safety
margin, then prints the linearly-scaled learning rate.

Run on the lab machine BEFORE starting training:
    python scripts/autotune_batch_size.py

Update densenet_config.yaml with the printed values before launching training.
"""

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch
import yaml

from src.utils.device import get_device
from src.utils.paths import find_repo_root


REFERENCE_BATCH_SIZE = 32
REFERENCE_LR         = 0.001
SAFETY_MARGIN        = 0.80   # use at most 80% of VRAM
MIN_BATCH            = 1
MAX_BATCH            = 512


def load_config():
    repo = find_repo_root()
    cfg_path = repo / "configs" / "densenet_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_dummy_model():
    """Build the DenseNet121 with the modified stem (same as training)."""
    from src.models.densenet_3d import build_densenet
    cfg = load_config()
    return build_densenet(cfg)


def probe_batch_size(model, device: torch.device, batch_size: int) -> bool:
    """
    Returns True if a forward+backward pass at this batch size fits in VRAM.
    Clears cache after each probe.
    """
    try:
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, 1, 32, 32, 32, device=device)
        model.train()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = out.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
        del x, out, loss
        torch.cuda.empty_cache()

        peak = torch.cuda.max_memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return peak <= total * SAFETY_MARGIN
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False


def binary_search_batch_size(model, device: torch.device) -> int:
    lo, hi = MIN_BATCH, MAX_BATCH
    best = MIN_BATCH

    print(f"Binary-searching batch size (safety margin {int(SAFETY_MARGIN*100)}%)...")
    while lo <= hi:
        mid = (lo + hi) // 2
        fits = probe_batch_size(model, device, mid)
        status = "OK" if fits else "OOM"
        print(f"  batch_size={mid:4d} → {status}")
        if fits:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def main():
    print("=" * 60)
    print("AUTOTUNE BATCH SIZE")
    print("=" * 60)

    device = get_device()
    if device.type != "cuda":
        print(f"Device is {device.type}, not CUDA — batch size tuning skipped.")
        print(f"Use the reference: batch_size={REFERENCE_BATCH_SIZE}, lr={REFERENCE_LR}")
        return

    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"Total VRAM: {total_vram:.1f} GB")
    print()

    print("Building model...")
    model = build_dummy_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f} M")
    print()

    best_bs = binary_search_batch_size(model, device)

    # Linear LR scaling rule
    scaled_lr = REFERENCE_LR * (best_bs / REFERENCE_BATCH_SIZE)

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print()
    print("=" * 60)
    print("RESULT — update configs/densenet_config.yaml with:")
    print(f"  batch_size: {best_bs}")
    print(f"  lr: {scaled_lr:.6f}  (linear scaling from bs={REFERENCE_BATCH_SIZE}, lr={REFERENCE_LR})")
    print()
    print(f"Peak VRAM at batch_size={best_bs}: ~{peak_mem:.2f} GB / {total_vram:.1f} GB")
    print("=" * 60)


if __name__ == "__main__":
    main()
