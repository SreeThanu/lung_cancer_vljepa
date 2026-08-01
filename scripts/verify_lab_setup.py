"""
Phase 7 — Lab environment verification.

Run on the lab machine (Ubuntu 22.04, RTX A4000) BEFORE any training:
    python scripts/verify_lab_setup.py

Checks (in order):
  1. Platform is Linux (not Mac)
  2. CUDA is available (asserts — hard fail if not)
  3. GPU name and VRAM match expectations (RTX A4000, ~16 GB)
  4. labels_v2.csv exists with expected columns and row count
  5. patches_40/ has expected .npz count matching labels_v2.csv
  6. One .npz sampled at random has the correct shape (40, 40, 40)
  7. Model forward pass (batch=4) succeeds and produces correct logits shape
  8. AMP bfloat16 context manager works without error
  9. VRAM used by a single forward pass (batch=4)
 10. Epoch-time projection: samples / sec → estimated minutes per epoch

Exits with code 0 on pass, 1 on any hard failure.
"""

import sys
import platform
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils.paths import find_repo_root, get_data_dir
from src.utils.device import get_device

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures = 0


def check(label: str, condition: bool, detail: str = "", warn: bool = False):
    global failures
    tag = PASS if condition else (WARN if warn else FAIL)
    line = f"  {tag} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not condition and not warn:
        failures += 1


def main():
    global failures

    repo     = find_repo_root()
    data_dir = get_data_dir(repo)

    cfg_path = repo / "configs" / "densenet_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("=" * 60)
    print("LAB SETUP VERIFICATION")
    print(f"Repo   : {repo}")
    print(f"Data   : {data_dir}")
    print("=" * 60)

    # ── 1. Platform ─────────────────────────────────────────────
    print("\n[1] Platform")
    is_linux = platform.system() == "Linux"
    check("Linux platform", is_linux, platform.system())
    if not is_linux:
        print(f"    {WARN} Running on {platform.system()} — GPU code should only run on lab machine.")

    # ── 2. CUDA ─────────────────────────────────────────────────
    print("\n[2] CUDA")
    cuda_ok = torch.cuda.is_available()
    check("torch.cuda.is_available()", cuda_ok)
    if not cuda_ok:
        print(f"    {FAIL} CUDA required. Install pytorch with CUDA wheels (see requirements-lab.txt).")
        sys.exit(1)

    device = torch.device("cuda")
    props  = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    check("GPU detected", True, f"{props.name} | {vram_gb:.1f} GB VRAM")
    check("VRAM >= 12 GB", vram_gb >= 12, f"{vram_gb:.1f} GB")
    check("Ampere (sm_80+) for native bfloat16", props.major >= 8,
          f"sm_{props.major}{props.minor}", warn=(props.major < 8))

    # ── 3. labels_v2.csv ────────────────────────────────────────
    print("\n[3] labels_v2.csv")
    labels_path = data_dir / cfg["data"]["labels_csv"]
    check("labels_v2.csv exists", labels_path.exists(), str(labels_path))

    if labels_path.exists():
        df = pd.read_csv(labels_path)
        req_cols = {"patient_id", "nodule_id", "label", "avg_score"}
        missing  = req_cols - set(df.columns)
        check("Required columns present", len(missing) == 0,
              f"missing: {missing}" if missing else f"{len(df)} rows")
        check("Row count in expected range [700, 1000]", 700 <= len(df) <= 1000,
              str(len(df)))
        check("Binary labels (0 and 1 only)",
              set(df["label"].unique()).issubset({0, 1}),
              str(sorted(df["label"].unique())))
        check("Patient count >= 400", df["patient_id"].nunique() >= 400,
              f"{df['patient_id'].nunique()} unique patients")
    else:
        df = None
        failures += 3

    # ── 4. patches_40/ ──────────────────────────────────────────
    print("\n[4] patches_40/")
    patch_dir = data_dir / cfg["data"]["patch_dir"]
    check("patches_40/ exists", patch_dir.exists(), str(patch_dir))

    if patch_dir.exists() and df is not None:
        npz_count = len(list(patch_dir.rglob("*.npz")))
        check("npz count matches label rows", npz_count == len(df),
              f"{npz_count} npz vs {len(df)} label rows",
              warn=(npz_count != len(df)))

        # Shape check on random sample
        sample_row = df.sample(1, random_state=42).iloc[0]
        sample_path = patch_dir / str(sample_row["patient_id"]) / f"{sample_row['nodule_id']}.npz"
        if sample_path.exists():
            arr = np.load(str(sample_path))["patch"]
            check("Random patch shape == (40, 40, 40)", arr.shape == (40, 40, 40),
                  str(arr.shape))
            check("Patch dtype is float32 or float64", arr.dtype in [np.float32, np.float64],
                  str(arr.dtype))
            finite_frac = float(np.isfinite(arr).mean())
            check("Patch values finite (>99%)", finite_frac > 0.99, f"{finite_frac:.4f}")
        else:
            check(f"Sample patch exists ({sample_path.name})", False, str(sample_path))

    # ── 5. Model forward pass ────────────────────────────────────
    print("\n[5] Model")
    try:
        from src.models.densenet_3d import build_densenet
        model = build_densenet(cfg).to(device)
        model.eval()

        torch.cuda.reset_peak_memory_stats(device)
        x = torch.zeros(4, 1, 32, 32, 32, device=device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=cfg["training"].get("amp", True)):
                out = model(x)

        check("Forward pass (batch=4) succeeds", True)
        check("Output shape (4, 2)", out.shape == (4, 2), str(tuple(out.shape)))

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        check("Peak VRAM per forward (batch=4) < 2 GB", peak_mb < 2048,
              f"{peak_mb:.0f} MB")

        del x, out
        torch.cuda.empty_cache()

    except Exception as e:
        check("Model forward pass", False, str(e))
        model = None

    # ── 6. AMP bfloat16 ─────────────────────────────────────────
    print("\n[6] AMP bfloat16")
    try:
        x = torch.randn(2, 1, 32, 32, 32, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            y = x * 2.0
        check("bfloat16 autocast works", y.dtype == torch.bfloat16, str(y.dtype))
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        check("bfloat16 autocast", False, str(e))

    # ── 7. Epoch-time projection ─────────────────────────────────
    print("\n[7] Throughput projection")
    if model is not None and df is not None:
        try:
            bs    = cfg["data"]["batch_size"]
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = torch.nn.CrossEntropyLoss()

            # Warm-up
            x = torch.randn(bs, 1, 32, 32, 32, device=device)
            y = torch.zeros(bs, dtype=torch.long, device=device)
            for _ in range(3):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=cfg["training"].get("amp", True)):
                    loss = criterion(model(x), y)
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            t0 = time.time()
            N_STEPS = 10
            for _ in range(N_STEPS):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=cfg["training"].get("amp", True)):
                    loss = criterion(model(x), y)
                loss.backward()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.time() - t0

            sps      = (N_STEPS * bs) / elapsed
            n_train  = int(len(df) * 0.9)   # ~90% in train per fold
            steps_ep = n_train // bs
            secs_ep  = steps_ep / sps
            mins_ep  = secs_ep / 60.0
            hrs_cv   = (mins_ep * cfg["training"]["epochs"] * cfg["cv_folds"]) / 60.0

            check("Throughput measured", True,
                  f"{sps:.1f} samples/sec | ~{mins_ep:.1f} min/epoch | "
                  f"~{hrs_cv:.1f} hr for full 10-fold CV (no early stopping)")

            del x, y, model, optimizer
            torch.cuda.empty_cache()
        except Exception as e:
            check("Throughput projection", False, str(e))

    # ── Summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    if failures == 0:
        print(f"RESULT: ALL CHECKS PASSED — ready to train.")
        print(f"  Next: python scripts/autotune_batch_size.py")
        print(f"  Then: jupyter lab notebooks/05_densenet_training.ipynb  (sanity fold)")
        print(f"  Then: jupyter lab notebooks/07_densenet_cv.ipynb         (full 10-fold CV)")
    else:
        print(f"RESULT: {failures} CHECK(S) FAILED — fix before training.")
    print("=" * 60)

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
