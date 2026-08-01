"""
Phase 3 — Center-crop 96³ patches to 40³ and package for lab transfer.

Runs on Mac (CPU/IO only).

For each nodule in labels_v2.csv:
  - Load its 96³ .npz patch (key 'patch', float16 in originals)
  - Center-crop to 40×40×40 (voxels 28:68 in each axis)
  - Save to data/processed/patches_40/<patient_id>/<nodule_id>.npz
    as float32, compressed, same 'patch' key
  - Assert output shape is exactly (40,40,40)

Rationale for 40³:
  Training will random-crop 40³ → 32³ (+/-4 voxel translation jitter using
  real tissue, not zero padding). Validation center-crops 40³ → 32³.
  This matches the published protocol exactly.

Zero-padding report:
  Detects boundary-clipped patches by checking fraction of zeros in the 40³
  crop AND checking if centroid_vox_z/y/x < 48 (within 48 voxels of vol edge).

Usage:
    python scripts/prepare_transfer.py \\
        --labels  <path-to-labels_v2.csv> \\
        --out-dir <path-to-patches_40-root>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.paths import find_repo_root, get_data_dir

CROP_IN  = 96
CROP_OUT = 40
CROP_START = (CROP_IN - CROP_OUT) // 2   # = 28
CROP_END   = CROP_START + CROP_OUT        # = 68
EDGE_MARGIN = CROP_IN // 2               # = 48 — within this many voxels of edge = padded


def parse_args():
    repo = find_repo_root()
    data = get_data_dir(repo)
    p = argparse.ArgumentParser(description="Phase 3: center-crop 96³→40³")
    p.add_argument("--labels",  default=str(data / "processed" / "labels_v2.csv"))
    p.add_argument("--out-dir", default=str(data / "processed" / "patches_40"))
    return p.parse_args()


def center_crop(arr: np.ndarray) -> np.ndarray:
    """Center-crop a (96,96,96) array to (40,40,40)."""
    return arr[CROP_START:CROP_END, CROP_START:CROP_END, CROP_START:CROP_END]


def main():
    args = parse_args()
    labels_path = Path(args.labels)
    out_root    = Path(args.out_dir)

    print("=" * 70)
    print("PHASE 3 — Center-crop 96³ → 40³ for transfer")
    print("=" * 70)
    print(f"  labels_v2.csv : {labels_path}")
    print(f"  output root   : {out_root}")
    print()

    df = pd.read_csv(labels_path)
    print(f"  Nodules to process: {len(df)}")
    print()

    out_root.mkdir(parents=True, exist_ok=True)

    boundary_cases = []     # (patient_id, nodule_id, frac_zeros, near_edge_axes)
    errors         = []     # (patient_id, nodule_id, error_msg)
    n_written      = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping patches"):
        patient_id = row["patient_id"]
        nodule_id  = str(row["nodule_id"])
        src_path   = Path(row["patch_path"])

        try:
            # Load original 96³ patch
            src = np.load(str(src_path))
            patch96 = src["patch"].astype(np.float32)

            if patch96.shape != (CROP_IN, CROP_IN, CROP_IN):
                raise ValueError(
                    f"Expected ({CROP_IN},{CROP_IN},{CROP_IN}), "
                    f"got {patch96.shape}"
                )

            # Center-crop to 40³
            patch40 = center_crop(patch96)

            assert patch40.shape == (CROP_OUT, CROP_OUT, CROP_OUT), (
                f"Crop produced {patch40.shape}, expected "
                f"({CROP_OUT},{CROP_OUT},{CROP_OUT})"
            )

            # Zero-fraction check on the 40³ crop
            frac_zeros = float(np.mean(patch40 == 0.0))

            # Boundary proximity check using centroid coordinates
            cvz = row.get("centroid_vox_z", None)
            cvy = row.get("centroid_vox_y", None)
            cvx = row.get("centroid_vox_x", None)
            near_edge_axes = []
            if pd.notna(cvz) and cvz < EDGE_MARGIN:
                near_edge_axes.append(f"z(={int(cvz)})")
            if pd.notna(cvy) and cvy < EDGE_MARGIN:
                near_edge_axes.append(f"y(={int(cvy)})")
            if pd.notna(cvx) and cvx < EDGE_MARGIN:
                near_edge_axes.append(f"x(={int(cvx)})")

            if near_edge_axes or frac_zeros > 0.01:
                boundary_cases.append({
                    "patient_id":    patient_id,
                    "nodule_id":     nodule_id,
                    "frac_zeros":    frac_zeros,
                    "near_edge_axes": ", ".join(near_edge_axes) if near_edge_axes else "none (detected via zeros)",
                })

            # Save
            patient_dir = out_root / patient_id
            patient_dir.mkdir(exist_ok=True)
            out_path = patient_dir / f"{nodule_id}.npz"
            np.savez_compressed(str(out_path), patch=patch40)

            n_written += 1

        except Exception as e:
            errors.append({
                "patient_id": patient_id,
                "nodule_id":  nodule_id,
                "error":      str(e),
            })

    print()
    print(f"Written: {n_written} / {len(df)} patches")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e['patient_id']} | {e['nodule_id']}: {e['error']}")
    else:
        print("Errors: 0")

    # ------------------------------------------------------------------
    # Boundary / zero-padding report
    # ------------------------------------------------------------------
    print()
    print(f"Boundary/zero-padding cases: {len(boundary_cases)}")
    if boundary_cases:
        print(f"  (Not dropped — reported only per protocol)")
        print(f"  {'patient_id':<20} {'nodule_id':<40} {'frac_zeros':>10} near_edge_axes")
        for bc in boundary_cases:
            nid = bc['nodule_id'][:38]
            print(f"  {bc['patient_id']:<20} {nid:<40} {bc['frac_zeros']:>10.3f} {bc['near_edge_axes']}")

    # ------------------------------------------------------------------
    # Disk usage
    # ------------------------------------------------------------------
    total_bytes = sum(
        f.stat().st_size
        for f in out_root.rglob("*.npz")
    )
    total_mb  = total_bytes / 1024**2
    total_gb  = total_bytes / 1024**3

    print()
    print(f"Output size on disk: {total_mb:.1f} MB ({total_gb:.3f} GB)")
    print(f"Output directory: {out_root}")
    print()
    print("=" * 70)
    print("PHASE 3 COMPLETE — ready to write TRANSFER_MANIFEST.md")
    print("=" * 70)

    return {
        "n_written":        n_written,
        "n_errors":         len(errors),
        "n_boundary":       len(boundary_cases),
        "total_mb":         total_mb,
        "out_root":         str(out_root),
    }


if __name__ == "__main__":
    main()
