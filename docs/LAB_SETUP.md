# Lab Machine Setup Guide

Hardware target: Ubuntu 22.04, Intel i9-14900K, 64 GB RAM, RTX A4000 (~16 GB VRAM, Ampere).

## 1. Clone the repo

```bash
git clone <repo-url> lung_cancer_vljepa
cd lung_cancer_vljepa
```

Or copy the repo from the Mac T7 SSD via rsync (see TRANSFER_MANIFEST.md).

## 2. Create environment

```bash
conda create -n lung python=3.10
conda activate lung
```

## 3. Install PyTorch (CUDA 12.1 wheel — must be first)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA RTX A4000
```

## 4. Install remaining dependencies

```bash
pip install -r requirements-lab.txt
```

## 5. Place data

Transfer from the Mac T7 SSD (see TRANSFER_MANIFEST.md for rsync commands):

```
data/
  processed/
    labels_v2.csv          # 807 nodules, binary labels
    patches_40/            # 40³ center-cropped patches (~83 MB total)
      <patient_id>/
        <nodule_id>.npz
```

**Do NOT** copy `patches/` (96³, 2.2 GB), `full_volumes/` (40 GB), or raw DICOM.

### Set LUNG_DATA_DIR (optional, if data is not inside the repo)

If you place data outside the repo root, set:
```bash
export LUNG_DATA_DIR=/path/to/data
```
All scripts resolve data paths via `get_data_dir()` in `src/utils/paths.py`.
If `LUNG_DATA_DIR` is not set, defaults to `<repo_root>/data`.

## 6. Verify the setup

```bash
python scripts/verify_lab_setup.py
```

All checks must pass (green `[PASS]`) before training. Fix any `[FAIL]` items.

Expected output (abbreviated):
```
[1] Platform
  [PASS] Linux platform
[2] CUDA
  [PASS] torch.cuda.is_available()
  [PASS] GPU detected — NVIDIA RTX A4000 | 16.1 GB VRAM
  [PASS] VRAM >= 12 GB
  [PASS] Ampere (sm_80+) for native bfloat16
[3] labels_v2.csv
  [PASS] labels_v2.csv exists
  [PASS] Required columns present — 807 rows
  [PASS] Row count in expected range [700, 1000]
  [PASS] Binary labels (0 and 1 only)
[4] patches_40/
  [PASS] patches_40/ exists
  [PASS] npz count matches label rows
  [PASS] Random patch shape == (40, 40, 40)
[5] Model
  [PASS] Forward pass (batch=4) succeeds
  [PASS] Output shape (4, 2)
[6] AMP bfloat16
  [PASS] bfloat16 autocast works
[7] Throughput projection
  [PASS] ~180 samples/sec | ~3.8 min/epoch | ~6.4 hr for full 10-fold CV

RESULT: ALL CHECKS PASSED — ready to train.
```

## 7. Auto-tune batch size

The default config uses `batch_size: 32`. Before training, find the largest
batch that fits in 80% of VRAM:

```bash
python scripts/autotune_batch_size.py
```

Update `configs/densenet_config.yaml` with the printed `batch_size` and `lr`.

## 8. Sanity fold (optional — 5-10 min)

Run one fold to confirm the full pipeline end-to-end:

```bash
jupyter lab notebooks/05_densenet_training.ipynb
```

Run all cells. If val AUC > 0.6 after epoch 1 and the run completes without
error, proceed to the full CV.

## 9. Full 10-fold CV

```bash
jupyter lab notebooks/07_densenet_cv.ipynb
```

Or launch headlessly and leave running overnight:

```bash
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=43200 \
    --output notebooks/07_densenet_cv_output.ipynb \
    notebooks/07_densenet_cv.ipynb
```

Checkpoints are saved to `checkpoints/densenet/fold_N.pth` after each fold.
Summary written to `experiments/densenet_cv_summary.txt`.

## 10. Radiomics (Phase 8 — disabled by default)

Radiomics extraction is implemented but disabled (`use_radiomics: false` in config).
To enable:

```bash
python scripts/extract_radiomics.py
```

Then set `use_radiomics: true` in `configs/densenet_config.yaml` and re-run the CV.
The scaler is fit on train folds only (asserted in the extraction script).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: monai` | `pip install monai==1.3.2` |
| `torch.cuda.is_available()` returns False | Check CUDA wheel: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| VRAM OOM during training | Re-run `autotune_batch_size.py`; reduce `batch_size` manually |
| `FileNotFoundError: labels_v2.csv` | Check `LUNG_DATA_DIR` env var or place data at `data/processed/` inside repo |
| `FileNotFoundError: N patches not found` | Transfer `patches_40/` from Mac; see TRANSFER_MANIFEST.md |
| AMP bfloat16 error | Ampere GPU required; for older GPUs set `amp_dtype: float16` in config |
