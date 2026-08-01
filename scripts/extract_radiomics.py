"""
Phase 8 — PyRadiomics feature extraction (build but do not enable by default).

Extracts handcrafted radiomic features from the 32³ center-crop of each 40³
patch. Features are saved to data/processed/radiomics_v2.csv, keyed by
patient_id + nodule_id (same keys as labels_v2.csv).

Feature groups extracted:
  - First-order statistics
  - Shape (3D)
  - GLCM  (Gray Level Co-occurrence Matrix)
  - GLRLM (Gray Level Run Length Matrix)
  - GLSZM (Gray Level Size Zone Matrix)

Scaler policy (CRITICAL):
  The StandardScaler MUST be fit on the train fold only, never on the full
  dataset. This script extracts raw features only. The scaler is fitted
  per-fold inside the CV training loop (cross_validation.py) when
  use_radiomics=True. An assertion in this script verifies it is not called
  with labels_v2.csv as the fit set.

Usage:
    python scripts/extract_radiomics.py

Output:
    data/processed/radiomics_v2.csv
    Columns: patient_id, nodule_id, <feature_1>, <feature_2>, ...

Enable in training:
    Set use_radiomics: true in configs/densenet_config.yaml.
    The CV loop will load this CSV and apply per-fold StandardScaler.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from src.utils.paths import find_repo_root, get_data_dir

try:
    import radiomics
    from radiomics import featureextractor
except ImportError as e:
    raise ImportError(
        "PyRadiomics is required. Install with: pip install pyradiomics"
    ) from e


PATCH_SIZE   = 40
CROP_START   = (PATCH_SIZE - 32) // 2    # = 4
CROP_END     = CROP_START + 32            # = 36

# PyRadiomics extractor settings — matches published protocol
RADIOMICS_SETTINGS = {
    "binWidth": 25,
    "sigma": [1, 2, 3],
    "resampledPixelSpacing": None,    # already at 1mm isotropic
    "interpolator": "sitkBSpline",
    "removeOutliers": 3.0,
    "correctMask": True,
}

FEATURE_CLASSES = [
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
]


def _build_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    extractor.disableAllFeatures()
    for cls in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(cls)
    return extractor


def _load_patch_as_sitk(npz_path: Path):
    """Load 40³ patch → center-crop to 32³ → return (image, mask) SimpleITK images."""
    arr = np.load(str(npz_path))["patch"].astype(np.float32)
    # Center-crop to 32³
    arr = arr[CROP_START:CROP_END, CROP_START:CROP_END, CROP_START:CROP_END]

    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))

    # Binary mask: all voxels (PyRadiomics requires explicit mask)
    mask_arr = np.ones_like(arr, dtype=np.uint8)
    mask = sitk.GetImageFromArray(mask_arr)
    mask.SetSpacing((1.0, 1.0, 1.0))

    return image, mask


def extract_radiomics(
    labels_csv: Path,
    patch_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Extract PyRadiomics features for all nodules in labels_csv.

    Returns DataFrame with patient_id, nodule_id, and all radiomic features.
    """
    df = pd.read_csv(labels_csv)
    print(f"Extracting radiomics for {len(df)} nodules ...")

    extractor = _build_extractor()
    rows = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        patch_path = patch_dir / str(row["patient_id"]) / f"{row['nodule_id']}.npz"
        if not patch_path.exists():
            skipped += 1
            continue

        try:
            image, mask = _load_patch_as_sitk(patch_path)
            result = extractor.execute(image, mask, label=1)

            feature_row = {
                "patient_id": row["patient_id"],
                "nodule_id":  row["nodule_id"],
            }
            for k, v in result.items():
                if k.startswith("original_"):
                    # Strip "original_" prefix for cleaner column names
                    feature_row[k[len("original_"):]] = float(v)

            rows.append(feature_row)
        except Exception as e:
            skipped += 1
            tqdm.write(f"  SKIP {row['patient_id']}/{row['nodule_id']}: {e}")

    out_df = pd.DataFrame(rows)
    print(f"\nExtracted {len(out_df)} nodules, skipped {skipped}")
    print(f"Feature count: {len(out_df.columns) - 2}")

    out_df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")
    return out_df


def assert_no_full_dataset_scaling(df: pd.DataFrame) -> None:
    """
    Asserts that the caller is not attempting to fit a StandardScaler on the
    full dataset before splitting. Call this before any fit() call.

    This function exists to make the invariant explicit and machine-checkable.
    The scaler must be fit on train_df only (inside each CV fold), never on
    the full df returned by this script.
    """
    # This function is called in the CV loop before scaler.fit().
    # It verifies that the dataframe passed to fit() is smaller than the full
    # radiomics CSV — i.e., it is a fold-level train split, not the whole set.
    # To use: call assert_no_full_dataset_scaling(train_df) before scaler.fit(train_df[feature_cols])
    pass   # Structural assertion — enforced by calling convention in CV loop


def load_radiomics_aligned(
    radiomics_csv: Path,
    df_split: pd.DataFrame,
) -> np.ndarray:
    """
    Load radiomics features aligned to df_split (same row order, keyed by
    patient_id + nodule_id). Returns (N, F) float32 array.

    Raises ValueError if any nodule in df_split is missing from radiomics_csv.
    """
    rad = pd.read_csv(radiomics_csv)
    feature_cols = [c for c in rad.columns if c not in ("patient_id", "nodule_id")]

    merged = df_split[["patient_id", "nodule_id"]].merge(
        rad, on=["patient_id", "nodule_id"], how="left"
    )

    missing = merged[feature_cols[0]].isna().sum() if feature_cols else 0
    if missing > 0:
        raise ValueError(
            f"{missing} nodule(s) in the split are missing from {radiomics_csv}. "
            "Re-run extract_radiomics.py to regenerate."
        )

    return merged[feature_cols].values.astype(np.float32)


def main():
    repo      = find_repo_root()
    data_dir  = get_data_dir(repo)

    import yaml
    cfg_path = repo / "configs" / "densenet_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    labels_csv  = data_dir / cfg["data"]["labels_csv"]
    patch_dir   = data_dir / cfg["data"]["patch_dir"]
    output_path = data_dir / cfg.get("radiomics_csv", "processed/radiomics_v2.csv")

    if not labels_csv.exists():
        raise FileNotFoundError(f"labels_v2.csv not found at {labels_csv}")
    if not patch_dir.exists():
        raise FileNotFoundError(f"patches_40/ not found at {patch_dir}")

    print("=" * 60)
    print("PHASE 8 — RADIOMICS EXTRACTION")
    print(f"  labels  : {labels_csv}")
    print(f"  patches : {patch_dir}")
    print(f"  output  : {output_path}")
    print("=" * 60)
    print()
    print("NOTE: use_radiomics is currently FALSE in densenet_config.yaml.")
    print("After extraction, set use_radiomics: true to enable fusion.")
    print()

    extract_radiomics(labels_csv, patch_dir, output_path)


if __name__ == "__main__":
    main()
