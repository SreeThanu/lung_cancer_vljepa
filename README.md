# CT-JEPA: Vision-Only JEPA for Lung Cancer Classification

A research-grade implementation of Joint Embedding Predictive Architecture (JEPA) adapted for 3D medical imaging, specifically for lung cancer classification on the LIDC-IDRI dataset.

## 🎯 Project Overview

This project implements a **latent prediction-based self-supervised learning approach** for 3D CT volumes. Unlike traditional masked autoencoders that reconstruct pixels, CT-JEPA predicts latent representations in an abstract feature space, enabling more efficient and semantically meaningful pretraining.

### Key Features

- ✅ **Self-supervised pretraining** on unlabeled CT scans
- ✅ **3D Vision Transformer** encoder with 3D patch embedding
- ✅ **Momentum-updated target encoder** (EMA)
- ✅ **3D block masking strategy** for volumetric data
- ✅ **Latent prediction** (NOT pixel reconstruction)
- ✅ **Fine-tuning** for lung cancer classification
- ✅ **Medical imaging metrics** (Sensitivity, Specificity, ROC-AUC)

---

## 🏗️ Architecture: CT-JEPA

### What is JEPA?

**Joint Embedding Predictive Architecture (JEPA)** is a self-supervised learning paradigm that learns representations by predicting the latent embeddings of masked regions rather than reconstructing pixels.

### Why JEPA for Medical Imaging?

1. **Efficiency**: Avoids expensive pixel-level reconstruction
2. **Semantic Focus**: Learns high-level features rather than low-level textures
3. **Data Efficiency**: Works well with limited labeled data
4. **Transferability**: Pretrained representations transfer well to downstream tasks

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input CT Volume [96³]                         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          │                            │
          ▼                            ▼
  ┌───────────────┐            ┌───────────────┐
  │   3D Block    │            │  Target       │
  │   Masking     │            │  Encoder      │
  │  (40% mask)   │            │  (EMA, no_grad)│
  └───────┬───────┘            └───────┬───────┘
          │                            │
          ▼                            ▼
  ┌───────────────┐            ┌───────────────┐
  │  Context      │            │ Target Latents│
  │  Encoder      │            │ [full volume] │
  │  (Student)    │            └───────────────┘
  │ [visible only]│                    │
  └───────┬───────┘                    │
          │                            │
          ▼                            │
  ┌───────────────┐                    │
  │  Predictor    │                    │
  │  Network      │                    │
  └───────┬───────┘                    │
          │                            │
          ▼                            ▼
  ┌─────────────────────────────────────┐
  │       Loss (MSE / Cosine)           │
  │   Predicted ↔ Target Latents        │
  └─────────────────────────────────────┘
           │
           ▼
    Backprop (Student + Predictor)
           │
           ▼
    EMA Update (Teacher)
```

### Key Components

1. **Context Encoder (Student)**
   - 3D Vision Transformer (ViT-3D)
   - Processes only visible patches
   - Trainable via backpropagation

2. **Target Encoder (Teacher)**
   - EMA copy of context encoder
   - Processes full volume
   - Updated via momentum (no gradients)

3. **Predictor**
   - Lightweight transformer
   - Maps context latents → target latents
   - Includes learnable mask tokens

4. **3D Block Masking**
   - Random contiguous cuboid masks
   - 30-50% masking ratio
   - Preserves spatial coherence

---

## 📊 Dataset: LIDC-IDRI

### Dataset Structure

```
LIDC-IDRI/
├── LIDC-IDRI-0001/
│   ├── <study_uid>/
│   │   └── <series_uid>/
│   │       ├── slice_001.dcm
│   │       ├── slice_002.dcm
│   │       └── ...
│   └── annotations.xml
├── LIDC-IDRI-0002/
└── ...
```

### Key Characteristics

- **Format**: DICOM CT scans
- **Annotations**: XML files with radiologist markings
- **Nodule Info**: Malignancy scores (1-5), size, spiculation
- **Multiple Readers**: Up to 4 radiologists per scan
- **Total Patients**: 1,018 patients

### Label Generation

1. Extract nodule annotations from XML
2. Compute consensus malignancy score
3. Binary classification: malignancy ≥ 4 → cancer (1), else benign (0)
4. Patient-level split: 70% train, 15% val, 15% test

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone <repo_url>
cd lung_cancer_vl_jepa_lidc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **Download LIDC-IDRI**
   - Register at [TCIA](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
   - Download dataset
   - Extract to `data/raw/LIDC-IDRI/`

2. **Run Preprocessing**
   ```bash
   # Option 1: Using notebook
   jupyter notebook notebooks/02_data_preprocessing.ipynb
   
   # Option 2: Using script
   python src/preprocessing/preprocess.py
   ```

---

## 📓 Notebook Execution Order

### 1. Data Exploration (`01_data_exploration.ipynb`)

- Visualize CT slices
- Analyze HU distribution
- Parse XML annotations
- Examine malignancy distribution

### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)

- DICOM → HU conversion
- Lung window clipping [-1000, 400]
- Normalize to [0, 1]
- Resample to 1mm isotropic
- Save as .npy files

### 3. Model Training (`03_model_training.ipynb`)

- Initialize CT-JEPA model
- Visualize masking strategy
- Pretrain on unlabeled CT volumes
- Save checkpoints

### 4. Model Evaluation (`04_model_evaluation.ipynb`)

- Load pretrained encoder
- Fine-tune on labeled data
- Compute metrics (Accuracy, ROC-AUC, Sensitivity, Specificity)
- Plot ROC curve and confusion matrix

---

## 🏋️ Training

### Self-Supervised Pretraining

```bash
python src/training/train_jepa.py
```

**Configuration** (`configs/jepa_config.yaml`):
- Epochs: 200
- Batch size: 4 (adjust based on GPU memory)
- Learning rate: 1e-4
- Optimizer: AdamW
- Mixed precision: Enabled
- Masking ratio: 40%

### Supervised Fine-tuning

```bash
python src/evaluation/evaluate.py
```

**Configuration**:
- Epochs: 50
- Learning rate: 5e-4
- Freeze encoder: False (full fine-tuning)
- Class weighting: Enabled (for imbalance)

---

## 📈 Expected Results

### Pretraining Loss

- Should decrease steadily over epochs
- Typical final loss: 0.01-0.05 (depends on loss function)

### Classification Metrics

Target performance (with sufficient data):
- **Accuracy**: 80-90%
- **ROC-AUC**: 0.85-0.95
- **Sensitivity**: 85-95% (critical for cancer detection)
- **Specificity**: 70-85%

---

## 🔬 Technical Details

### Preprocessing Pipeline

```python
CT Scan (DICOM)
    ↓
Convert to HU (slope * pixel + intercept)
    ↓
Clip to lung window [-1000, 400]
    ↓
Min-max normalize to [0, 1]
    ↓
Resample to isotropic 1mm spacing
    ↓
Extract 96×96×96 patches
    ↓
Save as .npy tensors
```

### Training Strategy

**Phase 1: Self-Supervised Pretraining**
- Input: Unlabeled CT volumes
- Loss: Latent prediction (MSE/Cosine)
- Updates: Context encoder + Predictor
- EMA: Target encoder (momentum 0.996 → 1.0)

**Phase 2: Supervised Fine-tuning**
- Input: Labeled CT volumes with cancer labels
- Loss: Cross-entropy (with focal loss option)
- Updates: Encoder + Classifier
- Techniques: Layer-wise LR decay, class balancing

---

## 📂 Project Structure

```
lung_cancer_vl_jepa_lidc/
│
├── configs/
│   └── jepa_config.yaml          # Configuration file
│
├── data/
│   ├── raw/LIDC-IDRI/           # Raw DICOM data
│   └── processed/               # Preprocessed volumes
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data_exploration/
│   │   └── exploration.py       # Data loading and visualization
│   ├── preprocessing/
│   │   └── preprocess.py        # CT preprocessing pipeline
│   ├── models/
│   │   ├── encoder_3d.py        # 3D ViT encoder
│   │   ├── momentum_encoder.py  # EMA target encoder
│   │   ├── jepa_predictor.py    # Predictor network
│   │   ├── masking.py           # 3D block masking
│   │   └── classification_head.py
│   ├── training/
│   │   └── train_jepa.py        # JEPA pretraining
│   ├── evaluation/
│   │   └── evaluate.py          # Classification evaluation
│   └── utils/
│       ├── config.py
│       ├── dataloader.py
│       ├── losses.py
│       ├── metrics.py
│       └── visualization.py
│
├── requirements.txt
└── README.md
```

---

## 🎓 Why This Approach Works

### Advantages of JEPA over Pixel Reconstruction

1. **Computational Efficiency**
   - No need to decode high-res 3D volumes
   - Predictor is lightweight compared to decoder

2. **Semantic Learning**
   - Forces model to understand abstract patterns
   - Ignores low-level noise and artifacts

3. **Better Representations**
   - Latent space predictions are more task-relevant
   - Transfers better to downstream classification

4. **Medical Imaging Benefits**
   - Less sensitive to scanner variations
   - Focuses on anatomical structures, not pixel intensities

---

## 🔮 Future Improvements

1. **Architecture Enhancements**
   - Experiment with Swin Transformer for efficiency
   - Multi-scale processing
   - Attention visualization

2. **Training Strategies**
   - Multi-task learning (size + malignancy)
   - Curriculum learning (easy → hard samples)
   - Contrastive loss variants

3. **Data Augmentation**
   - Advanced 3D augmentations (elastic deformation)
   - Mixup for medical imaging
   - Synthetic nodule injection

4. **Clinical Deployment**
   - Model compression (quantization, pruning)
   - Uncertainty estimation
   - Explainability (GradCAM for 3D)

---

## 📚 References

1. **JEPA**: LeCun, Y. "A Path Towards Autonomous Machine Intelligence" (2022)
2. **I-JEPA**: Assran et al. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture" (2023)
3. **LIDC-IDRI**: Armato III et al. "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)" (2011)
4. **Vision Transformers**: Dosovitskiy et al. "An Image is Worth 16x16 Words" (2021)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

This project is for research purposes. Please cite appropriately if used in publications.

---

## ⚠️ Disclaimer

This is a research prototype and should NOT be used for clinical diagnosis without proper validation and regulatory approval.

---

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Happy Research! 🚀**
