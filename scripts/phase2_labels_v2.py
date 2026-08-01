"""
Phase 2 — Regenerate labels under the new protocol.

Runs on Mac (CPU/IO only). Does NOT touch .npz patches.

Protocol:
  - Fix XML double-parsing (series-UID deduplication, already in preprocess.py)
  - min_radiologists >= 3
  - Option A label rule:
      benign    = avg_score <= 2.5   (label = 0)
      malignant = avg_score >= 3.5   (label = 1)
      DELETE    = 2.5 < avg_score < 3.5 (ambiguous, excluded entirely)
  - Drop any rows whose .npz patch is missing on disk
  - Write data/processed/labels_v2.csv
  - Leave labels.csv untouched

Usage:
    python scripts/phase2_labels_v2.py \\
        --raw-dir  <path-to-LIDC-IDRI-DICOM-root> \\
        --xml-dir  <path-to-LIDC-XML-only>         \\
        --labels   <path-to-existing-labels.csv>   \\
        --out      <path-for-labels_v2.csv>

    Or rely on LUNG_DATA_DIR:
        LUNG_DATA_DIR=/data python scripts/phase2_labels_v2.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.paths import find_repo_root, get_data_dir
from src.preprocessing.preprocess import (
    build_series_to_patient,
    parse_all_xml,
    aggregate_nodules,
)

SOFT_LABEL_MAP = {1: 0.0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1.0}
MIN_RADIOLOGISTS = 3
AMBIGUOUS_LO = 2.5
AMBIGUOUS_HI = 3.5


def parse_args():
    repo = find_repo_root()
    data = get_data_dir(repo)

    p = argparse.ArgumentParser(description="Phase 2: regenerate labels_v2.csv")
    p.add_argument("--raw-dir",  default=str(data / "raw" / "LIDC-IDRI"))
    p.add_argument("--xml-dir",  default=str(data / "raw" / "annotations" / "LIDC-XML-only"))
    p.add_argument("--labels",   default=str(data / "processed" / "labels.csv"))
    p.add_argument("--out",      default=str(data / "processed" / "labels_v2.csv"))
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PHASE 2 — Regenerate labels under new protocol")
    print("=" * 70)
    print(f"  raw_dir : {args.raw_dir}")
    print(f"  xml_dir : {args.xml_dir}")
    print(f"  labels  : {args.labels}")
    print(f"  out     : {args.out}")
    print()

    # ------------------------------------------------------------------
    # Reload old labels.csv — needed for centroid/patch_path columns
    # ------------------------------------------------------------------
    old_df = pd.read_csv(args.labels)
    step0 = len(old_df)
    print(f"Filter cascade:")
    print(f"  starting nodules (labels.csv)          : {step0}")

    # ------------------------------------------------------------------
    # Step 1: series→patient map (fast, ~2 sec)
    # ------------------------------------------------------------------
    series_to_patient = build_series_to_patient(args.raw_dir)

    # ------------------------------------------------------------------
    # Step 2 & 3: parse XMLs (with dedup fix) + aggregate (min_rad=2)
    # ------------------------------------------------------------------
    raw_index = parse_all_xml(args.xml_dir, series_to_patient)
    # aggregate at min_rad=2 first so we can track the cascade correctly
    patient_annots_2 = aggregate_nodules(raw_index, 2, SOFT_LABEL_MAP)

    rows_after_dedup = []
    for pid, nodules in patient_annots_2.items():
        for nd in nodules:
            rows_after_dedup.append({
                "patient_id":       pid,
                "nodule_id":        nd["nodule_id"],
                "avg_score":        nd["avg_score"],
                "soft_label":       nd["soft_label"],
                "num_radiologists": nd["num_radiologists"],
            })
    dedup_df = pd.DataFrame(rows_after_dedup)
    step1 = len(dedup_df)
    print(f"  -> after double-parse fix              : {step1}  (removed {step0 - step1})")

    # ------------------------------------------------------------------
    # Apply min_radiologists >= 3
    # ------------------------------------------------------------------
    step2_df = dedup_df[dedup_df["num_radiologists"] >= MIN_RADIOLOGISTS].copy()
    step2 = len(step2_df)
    print(f"  -> after min_radiologists >= 3         : {step2}  (removed {step1 - step2})")

    # ------------------------------------------------------------------
    # Apply Option A: delete ambiguous (2.5 < avg_score < 3.5)
    # ------------------------------------------------------------------
    not_ambiguous = (step2_df["avg_score"] <= AMBIGUOUS_LO) | \
                    (step2_df["avg_score"] >= AMBIGUOUS_HI)
    step3_df = step2_df[not_ambiguous].copy()
    step3 = len(step3_df)
    print(f"  -> after deleting ambiguous (2.5,3.5)  : {step3}  (removed {step2 - step3})")

    # Assign binary label
    step3_df["label"] = np.where(step3_df["avg_score"] <= AMBIGUOUS_LO, 0, 1)

    # ------------------------------------------------------------------
    # Join with old labels.csv to get centroid/patch_path columns
    # ------------------------------------------------------------------
    keep_cols = [
        "patient_id", "nodule_id",
        "centroid_z_world", "centroid_y_px", "centroid_x_px",
        "centroid_vox_z",   "centroid_vox_y", "centroid_vox_x",
        "patch_shape", "patch_path",
    ]
    merged = step3_df.merge(
        old_df[keep_cols],
        on=["patient_id", "nodule_id"],
        how="left",
    )

    # Nodules in new set but missing from old (shouldn't happen)
    missing_in_old = merged["patch_path"].isna()
    if missing_in_old.any():
        print()
        print(f"  WARNING: {missing_in_old.sum()} nodules have no entry in old labels.csv "
              "(no patch_path). These cannot be verified on disk and will be dropped.")
        merged = merged[~missing_in_old].copy()

    # ------------------------------------------------------------------
    # Drop rows whose .npz is missing on disk
    # ------------------------------------------------------------------
    exists_on_disk = merged["patch_path"].apply(os.path.exists)
    missing_on_disk = merged[~exists_on_disk]
    step4_df = merged[exists_on_disk].copy()
    step4 = len(step4_df)
    print(f"  -> after dropping missing .npz         : {step4}  (removed {step3 - step4})")

    if len(missing_on_disk) > 0:
        print()
        print("  Nodules referencing missing .npz files:")
        for _, row in missing_on_disk.iterrows():
            print(f"    {row['patient_id']} | {row['nodule_id']} | {row['patch_path']}")

    print(f"  -> FINAL                               : {step4}")
    print()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    benign_mask = step4_df["label"] == 0
    malignant_mask = step4_df["label"] == 1
    n_benign = benign_mask.sum()
    n_malignant = malignant_mask.sum()
    total = len(step4_df)

    print("Final dataset summary:")
    print(f"  Total nodules   : {total}")
    print(f"  Benign  (0)     : {n_benign}  ({100*n_benign/total:.1f}%)")
    print(f"  Malignant (1)   : {n_malignant}  ({100*n_malignant/total:.1f}%)")
    print()

    unique_pids = step4_df["patient_id"].nunique()
    print(f"  Unique patients : {unique_pids}")
    print()

    npp = step4_df.groupby("patient_id").size()
    print("  Nodules-per-patient distribution:")
    npp_dist = npp.value_counts().sort_index()
    for k, v in npp_dist.items():
        print(f"    {k} nodule(s): {v} patient(s)")
    n_single = (npp == 1).sum()
    print(f"  Patients with exactly 1 nodule: {n_single}")
    print()

    # ------------------------------------------------------------------
    # Write labels_v2.csv
    # ------------------------------------------------------------------
    out_cols = [
        "patient_id", "nodule_id",
        "label", "avg_score", "soft_label", "num_radiologists",
        "centroid_z_world", "centroid_y_px", "centroid_x_px",
        "centroid_vox_z",   "centroid_vox_y", "centroid_vox_x",
        "patch_shape", "patch_path",
    ]
    step4_df[out_cols].to_csv(args.out, index=False)
    print(f"Wrote {step4} rows to {args.out}")
    print()
    print("=" * 70)
    print("PHASE 2 COMPLETE — labels_v2.csv written. Review then run Phase 3.")
    print("=" * 70)


if __name__ == "__main__":
    main()
