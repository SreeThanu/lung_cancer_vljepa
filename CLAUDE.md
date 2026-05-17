# CT-JEPA: Lung Cancer Classification Project
## Claude Code Project Instructions

---

## Project Overview

This project implements **CT-JEPA**, a self-supervised 3D Vision Transformer using Joint Embedding Predictive Architecture (JEPA) for lung cancer classification on the LIDC-IDRI dataset.

The core idea: pretrain a 3D ViT encoder using JEPA self-supervision on all available CT volumes (no labels needed), then fine-tune for binary lung cancer classification using soft labels derived from radiologist malignancy scores.

---

## Directory Structure

```
/Volumes/thanu's T7/lung_cancer_vljepa/
│
├── code/                          ← cloned GitHub repo (work here)
│   ├── CLAUDE.md                  ← this file
│   ├── configs/
│   │   └── jepa_config.yaml       ← all hyperparameters live here
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_data_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_model_evaluation.ipynb
│   ├── src/
│   │   ├── data_exploration/
│   │   │   └── exploration.py
│   │   ├── preprocessing/
│   │   │   └── preprocess.py      ← main preprocessing pipeline
│   │   ├── models/
│   │   │   ├── encoder_3d.py      ← 3D ViT encoder
│   │   │   ├── momentum_encoder.py ← EMA target encoder
│   │   │   ├── jepa_predictor.py  ← predictor network
│   │   │   ├── masking.py         ← 3D block masking
│   │   │   └── classification_head.py
│   │   ├── training/
│   │   │   └── train_jepa.py      ← JEPA pretraining
│   │   ├── evaluation/
│   │   │   └── evaluate.py        ← classification evaluation
│   │   └── utils/
│   │       ├── config.py
│   │       ├── dataloader.py
│   │       ├── losses.py
│   │       ├── metrics.py
│   │       └── visualization.py
│   ├── requirements.txt
│   └── README.md
│
├── data/
│   ├── raw/
│   │   └── LIDC-IDRI/             ← 1018 patient CT scans (DICOM format, ~124GB)
│   │       ├── 1.3.6.1.4.1.14519.5.2.1.6279.6001.XXX/  ← series folders
│   │       └── ...
│   └── processed/                 ← preprocessed .npz patches (output here)
│       ├── patches/               ← nodule-centered 96x96x96 patches
│       ├── full_volumes/          ← resampled full CT volumes for pretraining
│       ├── metadata.csv           ← per-patch metadata
│       └── labels.csv             ← soft labels per nodule
│
├── checkpoints/                   ← model checkpoints
│   ├── pretrain/                  ← JEPA pretraining checkpoints
│   └── finetune/                  ← classification fine-tuning checkpoints
│
└── experiments/                   ← logs, results, plots
    └── lung_cancer/
```

---

## Data

### Raw Data Location
```
/Volumes/thanu's T7/lung_cancer_vljepa/data/raw/LIDC-IDRI/
```

### Dataset Facts
- **Total patients**: 1,018
- **Format**: DICOM CT scans organized by SeriesInstanceUID folders
- **Annotations**: XML files with radiologist nodule markings (up to 4 radiologists per scan)
- **Malignancy scores**: 1-5 per radiologist per nodule
- **Total series folders**: ~1,302 (multiple series per patient)

### Critical Dataset Issues (understand these before touching preprocessing)
1. **Annotation inconsistency**: Up to 4 radiologists annotate each scan, often disagreeing
2. **Score ambiguity**: Score 3 = genuinely uncertain (largest group, ~438 nodules)
3. **Missing XML**: Some patients have incomplete or missing annotation files
4. **Mixed coordinate formats**: XML gives imageZposition in world mm but xCoord/yCoord as pixel indices — NOT world coordinates (this caused major bugs before)
5. **Variable DICOM quality**: Some series have inconsistent slice spacing, missing slices, or corrupt files
6. **Small nodules**: Many nodules <3mm that are hard to localize in 3D patches

---

## Label Strategy

### DO NOT use hard binary labels with a simple threshold.

### Use soft labels based on average malignancy score:

```python
# Soft label mapping
label_map = {
    1: 0.0,   # definitely benign
    2: 0.1,   # probably benign
    3: 0.5,   # uncertain — keep but use 0.5
    4: 0.9,   # probably malignant
    5: 1.0    # definitely malignant
}

# Average across all radiologists who annotated the nodule
# Then map to soft label using above
```

### For evaluation only (not training):
- Use hard labels: scores 1-2 → benign (0), scores 4-5 → malignant (1)
- Exclude score-3 nodules from evaluation metrics
- Score-3 nodules are used in training but not evaluation

### Rationale
Previous approach dropped ~740 patients by filtering aggressively. This time we keep ALL patients. Score-3 nodules contribute to JEPA pretraining (no labels needed) and to fine-tuning via soft labels (preserving uncertainty). This is supported by Zhang et al. 2022 "Re-thinking and Re-labeling LIDC-IDRI".

---

## Preprocessing Philosophy

### The Previous Mistakes (DO NOT REPEAT)
1. ❌ Dropped patients if XML was missing or incomplete
2. ❌ Required ALL 4 radiologists to annotate a nodule
3. ❌ Discarded patients where patch extraction failed
4. ❌ Hard filter on malignancy score (excluded score 3)
5. ❌ Raised ValueError and dropped entire patients on any error
6. ❌ Treated xCoord/yCoord from XML as world mm coordinates (they are pixel indices)
7. ❌ Only preprocessed nodule patches — didn't save full volumes for pretraining

### The New Approach
1. ✅ Keep patients even with partial annotations — use what's available
2. ✅ Accept nodules annotated by at least 2 radiologists (not all 4)
3. ✅ Log failed patches but don't drop the patient — skip the nodule, keep the scan
4. ✅ Include score-3 nodules with soft label 0.5
5. ✅ Use try/except everywhere — errors are logged, not crashes
6. ✅ Correct coordinate conversion: xCoord/yCoord are pixel indices, imageZposition is world mm
7. ✅ Save full resampled volumes for JEPA pretraining (not just patches)

### Coordinate Bug Fix (CRITICAL)
```python
# WRONG (old approach — treats pixel coords as world mm):
centroid_world = (z_from_xml, y_from_xml, x_from_xml)  # BUG

# CORRECT (new approach):
x_world = origin[0] + x_pixel * spacing[0]  # pixel → world mm
y_world = origin[1] + y_pixel * spacing[1]  # pixel → world mm
z_world = z_from_xml                          # already world mm
```

### Preprocessing Pipeline (Two Outputs)

**Output 1: Full volumes** (for JEPA pretraining)
- All 1,018 patients
- Resampled to 1mm isotropic spacing
- HU clipped to [-1000, 400] and normalized to [0, 1]
- Saved as .npz compressed arrays
- No labels needed

**Output 2: Nodule patches** (for fine-tuning)
- 96×96×96 patches centered on annotated nodules
- Soft labels from malignancy score mapping
- From as many patients as possible (target: 700+)
- Saved as .npz with patch + label + metadata

---

## Training Strategy

### Phase 1: JEPA Self-Supervised Pretraining
- **Data**: All 1,018 patients (full volumes, no labels)
- **Model**: 3D ViT encoder + predictor + EMA target encoder
- **Loss**: MSE between predicted and target latents
- **Masking**: 3D block masking, 40% ratio
- **Epochs**: 200
- **Batch size**: 4 (adjust for M2 MacBook Pro memory)
- **Device**: MPS (Apple Silicon) — use `torch.device("mps")`

### Phase 2: Supervised Fine-tuning
- **Data**: Nodule patches with soft labels (700+ patients target)
- **Loss**: MSE loss (for soft labels) or BCE with label smoothing
- **Freeze encoder**: False (full fine-tuning)
- **Epochs**: 50
- **Evaluation**: Hard labels on scores 1,2,4,5 only

---

## Environment Setup

### Create a fresh conda environment:
```bash
conda create -n lung_cancer_v2 python=3.10
conda activate lung_cancer_v2
pip install -r requirements.txt
```

### Key libraries (from requirements.txt + additions as needed):
- `torch` with MPS support
- `SimpleITK` — DICOM loading and resampling
- `pylidc` — LIDC annotation parsing (preferred over manual XML)
- `numpy`, `pandas`, `tqdm`
- `scikit-learn` — metrics
- `matplotlib`, `seaborn` — visualization

### Device setup for M2 MacBook Pro:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Critical Rules for Claude Code

### Always follow these:
1. **Never crash on a single patient failure** — use try/except, log errors to a CSV, continue
2. **Always log what was skipped and why** — maintain a `skipped_patients.csv`
3. **Save checkpoints frequently** — every 10 epochs minimum
4. **Never hardcode paths** — always use the config file or pass as arguments
5. **Always use tqdm** for progress bars on long loops
6. **Test on 5 patients first** before running full pipeline
7. **Check MPS availability** before assuming GPU

### File paths to always use:
```python
RAW_DATA_DIR = "/Volumes/thanu's T7/lung_cancer_vljepa/data/raw/LIDC-IDRI"
PROCESSED_DIR = "/Volumes/thanu's T7/lung_cancer_vljepa/data/processed"
CHECKPOINT_DIR = "/Volumes/thanu's T7/lung_cancer_vljepa/checkpoints"
EXPERIMENT_DIR = "/Volumes/thanu's T7/lung_cancer_vljepa/experiments"
```

### When modifying preprocessing:
- Always keep the old version — never overwrite, create new versions
- Run on 10 patients first and verify output shapes before full run
- Log the number of patients processed vs skipped after every run

### When modifying models:
- Keep the config in `jepa_config.yaml` — never hardcode hyperparameters
- Always verify tensor shapes with print statements during first run
- MPS does not support all PyTorch operations — fall back to CPU for unsupported ops

---

## Known Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Only 278/1018 patients | Overly aggressive filtering | Tolerant preprocessing, keep score-3 |
| Centroid displacement | xCoord treated as world mm | Convert pixel → world using origin + spacing |
| Underfitting | Not enough pretraining data | Use all 1018 for pretraining |
| MPS errors | Unsupported ops on Apple Silicon | Add CPU fallback for specific ops |
| DICOM load failure | Corrupt/incomplete series | Try/except, log and skip |

---

## Research Context

### Architecture: CT-JEPA
- Based on I-JEPA (Assran et al. 2023)
- Adapted for 3D medical imaging
- Predicts latent representations (not pixels)
- More efficient than masked autoencoders for medical imaging

### Key Papers
1. LeCun 2022 — JEPA conceptual framework
2. Assran et al. 2023 — I-JEPA implementation
3. Zhang et al. 2022 — LIDC re-labeling and annotation issues
4. Dosovitskiy et al. 2021 — Vision Transformers

### Label Assignment Reference (Zhang et al. 2022)
- Scenario E performs best: scores 1-2 benign, 4-5 malignant, score-3 → benign
- But for our approach: soft labels preserve uncertainty better than hard assignment
- Score-3 as 0.5 is more honest than forcing benign/malignant

---

## Current Status
- [x] Dataset downloaded (1,018 CT series, ~124GB)
- [x] Raw data at `/Volumes/thanu's T7/lung_cancer_vljepa/data/raw/LIDC-IDRI`
- [ ] Fresh conda environment set up
- [ ] New preprocessing pipeline (tolerant version)
- [ ] Full volume extraction for pretraining
- [ ] Nodule patch extraction with soft labels
- [ ] JEPA pretraining on all 1,018 patients
- [ ] Fine-tuning on labeled subset
- [ ] Evaluation on clean label subset (scores 1,2,4,5)
