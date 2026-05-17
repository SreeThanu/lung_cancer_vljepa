"""Full preprocessing pipeline runner — logs to data/processed/pipeline_run.log"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.preprocessing.preprocess import run_pipeline
import pandas as pd
import numpy as np

config = load_config("configs/jepa_config.yaml")
cfg = config.data
PROCESSED = Path(cfg["data"]["processed_dir"])

print("=" * 50)
print("CT-JEPA FULL PREPROCESSING PIPELINE")
print("=" * 50)

meta_df, labels_df = run_pipeline(config, test_mode=False)

# ── Final summary ──────────────────────────────────────────────
vol_files   = list((PROCESSED / "full_volumes").glob("*.npz"))
patch_files = list((PROCESSED / "patches").rglob("*.npz"))
skipped_df  = pd.read_csv(PROCESSED / "skipped_patients.csv")

patient_skips = skipped_df[skipped_df["level"] == "patient"] if "level" in skipped_df.columns else skipped_df
nodule_skips  = skipped_df[skipped_df["level"] == "nodule"]  if "level" in skipped_df.columns else pd.DataFrame()

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Full volumes saved:       {len(vol_files)}")
print(f"Nodule patches saved:     {len(patch_files)}")
print(f"Patients skipped:         {len(patient_skips)}")
print(f"Nodule-level skips:       {len(nodule_skips)}")

if len(labels_df) > 0:
    print(f"\nLabel distribution ({len(labels_df)} nodules):")
    desc = labels_df["soft_label"].describe()
    print(f"  count  {int(desc['count'])}")
    print(f"  mean   {desc['mean']:.4f}")
    print(f"  std    {desc['std']:.4f}")
    print(f"  min    {desc['min']:.4f}")
    print(f"  25%    {desc['25%']:.4f}")
    print(f"  50%    {desc['50%']:.4f}")
    print(f"  75%    {desc['75%']:.4f}")
    print(f"  max    {desc['max']:.4f}")

    bins = [0.0, 0.15, 0.45, 0.55, 0.85, 1.01]
    labels_names = ["benign(0.0-0.15)", "prob_benign(0.15-0.45)", "uncertain(0.45-0.55)", "prob_malignant(0.55-0.85)", "malignant(0.85-1.0)"]
    cuts = pd.cut(labels_df["soft_label"], bins=bins, labels=labels_names, include_lowest=True)
    print(f"\nSoft label buckets:")
    for label, count in cuts.value_counts().sort_index().items():
        print(f"  {label}: {count}")

# Save summary JSON
summary = {
    "full_volumes": len(vol_files),
    "patches": len(patch_files),
    "patients_skipped": len(patient_skips),
    "nodule_skips": len(nodule_skips),
    "label_mean": float(labels_df["soft_label"].mean()) if len(labels_df) > 0 else None,
    "label_std":  float(labels_df["soft_label"].std())  if len(labels_df) > 0 else None,
}
with open(PROCESSED / "pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to {PROCESSED / 'pipeline_summary.json'}")
print("PIPELINE COMPLETE.")
