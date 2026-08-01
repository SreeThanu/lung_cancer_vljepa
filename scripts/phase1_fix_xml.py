"""
Phase 1 — Fix XML double-parsing bug and re-run XML annotation parsing.

Runs on Mac (CPU/IO only). Does NOT touch .npz patches.

What this script does:
  1. Builds series-UID → patient-ID mapping from DICOM headers (fast, one
     header per series directory).
  2. Runs the FIXED parse_all_xml, which deduplicates XML files by
     SeriesInstanceUid before parsing.
  3. Runs aggregate_nodules with the same min_radiologists=2 used originally,
     so the only change is the XML deduplication.
  4. Compares the new nodule records against the existing labels.csv:
     - Which nodules changed num_radiologists?
     - Which nodules changed avg_score? (if any: audit assumption was wrong)
  5. Reports the full distribution of num_radiologists — every value must be
     in {1, 2, 3, 4}. Any value > 4 means the fix is incomplete.

Usage:
    python scripts/phase1_fix_xml.py \
        --raw-dir  <path-to-LIDC-IDRI-DICOM-root> \
        --xml-dir  <path-to-LIDC-XML-only>        \
        --labels   <path-to-existing-labels.csv>

    Or with LUNG_DATA_DIR:
        LUNG_DATA_DIR=/data python scripts/phase1_fix_xml.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from anywhere in the repo
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.paths import find_repo_root, get_data_dir
from src.preprocessing.preprocess import (
    build_series_to_patient,
    parse_all_xml,
    aggregate_nodules,
)


SOFT_LABEL_MAP = {1: 0.0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1.0}


def parse_args():
    repo = find_repo_root()
    data = get_data_dir(repo)

    p = argparse.ArgumentParser(description="Phase 1: fix XML double-parsing")
    p.add_argument(
        "--raw-dir",
        default=str(data / "raw" / "LIDC-IDRI"),
        help="Root directory of LIDC-IDRI DICOM series folders",
    )
    p.add_argument(
        "--xml-dir",
        default=str(data / "raw" / "annotations" / "LIDC-XML-only"),
        help="Root directory of LIDC annotation XML package",
    )
    p.add_argument(
        "--labels",
        default=str(data / "processed" / "labels.csv"),
        help="Existing labels.csv to compare against",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PHASE 1 — XML double-parsing fix")
    print("=" * 70)
    print(f"  raw_dir : {args.raw_dir}")
    print(f"  xml_dir : {args.xml_dir}")
    print(f"  labels  : {args.labels}")
    print()

    # ------------------------------------------------------------------
    # Step 1: build series-UID → patient-ID mapping from DICOM headers
    # ------------------------------------------------------------------
    print("Step 1: Building series→patient map from DICOM headers ...")
    series_to_patient = build_series_to_patient(args.raw_dir)
    n_patients = len(set(series_to_patient.values()))
    print(f"  {len(series_to_patient)} series → {n_patients} unique patients")
    print()

    # ------------------------------------------------------------------
    # Step 2: parse XMLs with deduplication fix
    # ------------------------------------------------------------------
    print("Step 2: Parsing XMLs (with duplicate-series-UID fix) ...")
    raw_index = parse_all_xml(args.xml_dir, series_to_patient)
    total_raw = sum(len(v) for v in raw_index.values())
    print(f"  Parsed annotations for {len(raw_index)} patients, "
          f"{total_raw} total raw nodule entries")
    print()

    # ------------------------------------------------------------------
    # Step 3: aggregate nodules (same params as original run)
    # ------------------------------------------------------------------
    print("Step 3: Aggregating nodules (min_radiologists=2) ...")
    patient_annots = aggregate_nodules(
        raw_index,
        min_radiologists=2,
        soft_label_map=SOFT_LABEL_MAP,
    )
    total_nodules = sum(len(v) for v in patient_annots.values())
    print(f"  {len(patient_annots)} patients, {total_nodules} nodules")
    print()

    # Build a flat DataFrame from the aggregated result
    rows = []
    for patient_id, nodules in patient_annots.items():
        for nd in nodules:
            rows.append({
                "patient_id": patient_id,
                "nodule_id": nd["nodule_id"],
                "avg_score": nd["avg_score"],
                "soft_label": nd["soft_label"],
                "num_radiologists": nd["num_radiologists"],
            })
    new_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 4: compare with existing labels.csv
    # ------------------------------------------------------------------
    print("Step 4: Comparing with existing labels.csv ...")
    old_df = pd.read_csv(args.labels)
    print(f"  Old labels.csv: {len(old_df)} rows")
    print(f"  New aggregation: {len(new_df)} rows")
    print()

    # Merge on patient_id + nodule_id
    merged = old_df.merge(
        new_df[["patient_id", "nodule_id", "avg_score", "num_radiologists", "soft_label"]],
        on=["patient_id", "nodule_id"],
        suffixes=("_old", "_new"),
        how="outer",
        indicator=True,
    )

    both = merged[merged["_merge"] == "both"]
    only_old = merged[merged["_merge"] == "left_only"]
    only_new = merged[merged["_merge"] == "right_only"]

    print(f"  Matched (patient_id + nodule_id): {len(both)}")
    print(f"  Only in OLD labels.csv          : {len(only_old)}")
    print(f"  Only in NEW aggregation         : {len(only_new)}")
    print()

    # Nodules where num_radiologists changed
    rad_changed = both[both["num_radiologists_old"] != both["num_radiologists_new"]]
    print(f"  Nodules where num_radiologists changed: {len(rad_changed)}")
    if len(rad_changed) > 0:
        print("  Before → After:")
        val_counts = (
            rad_changed
            .groupby(["num_radiologists_old", "num_radiologists_new"])
            .size()
            .reset_index(name="count")
        )
        print(val_counts.to_string(index=False))
    print()

    # Nodules where avg_score changed (CRITICAL — audit assumed no change)
    score_changed = both[
        np.abs(both["avg_score_old"] - both["avg_score_new"]) > 1e-6
    ]
    print(f"  Nodules where avg_score changed: {len(score_changed)}")
    if len(score_changed) > 0:
        print()
        print("  *** AUDIT ASSUMPTION VIOLATED — avg_score DID CHANGE ***")
        print("  Showing all changed rows:")
        cols = ["patient_id", "nodule_id",
                "avg_score_old", "avg_score_new",
                "num_radiologists_old", "num_radiologists_new"]
        print(score_changed[cols].to_string(index=False))
    print()

    # ------------------------------------------------------------------
    # Step 5: num_radiologists distribution in new aggregation
    # ------------------------------------------------------------------
    print("Step 5: num_radiologists distribution (new aggregation) ...")
    dist = new_df["num_radiologists"].value_counts().sort_index()
    print(dist.to_string())
    print()

    bad = new_df[new_df["num_radiologists"] > 4]
    if len(bad) > 0:
        print(f"  *** STILL {len(bad)} nodules with num_radiologists > 4 ***")
        print("  Fix is INCOMPLETE. Investigate further before proceeding.")
        print()
        print(bad[["patient_id", "nodule_id", "num_radiologists", "avg_score"]].to_string())
    else:
        print("  All num_radiologists values in {1, 2, 3, 4}. Fix verified.")
    print()

    print("=" * 70)
    print("PHASE 1 COMPLETE — review results above before running Phase 2")
    print("=" * 70)


if __name__ == "__main__":
    main()
