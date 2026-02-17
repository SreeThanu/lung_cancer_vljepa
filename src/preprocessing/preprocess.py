"""
CT Scan Preprocessing Pipeline for LIDC-IDRI
VL-JEPA Compatible (Raw Patch Extraction Only)

Pipeline:
1. Load DICOM series
2. Convert to Hounsfield Units (HU)
3. Resample to isotropic spacing
4. Lung window clipping [-1000, 400]
5. Normalize to [0, 1]
6. Extract random 3D patches
7. Save compressed float16 patches

IMPORTANT:
- No context/target generation
- No masking
- Masking will be done dynamically during JEPA training
"""

import os
import gc
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm


# ================================================================
# CT PREPROCESSOR
# ================================================================

class CTPreprocessor:

    def __init__(
        self,
        clip_min: int = -1000,
        clip_max: int = 400,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        num_patches: int = 3
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.num_patches = num_patches

    # ------------------------------------------------------------
    # 1. Load DICOM Series
    # ------------------------------------------------------------
    def load_dicom_series(self, dicom_dir: str) -> sitk.Image:

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        return image

    # ------------------------------------------------------------
    # 2. Convert to Hounsfield Units
    # ------------------------------------------------------------
    def convert_to_hu(self, image: sitk.Image) -> np.ndarray:

        volume = sitk.GetArrayFromImage(image).astype(np.int16)

        intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0
        slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0

        volume = volume * slope + intercept

        return volume.astype(np.float32)

    # ------------------------------------------------------------
    # 3. Resample to Isotropic Spacing
    # ------------------------------------------------------------
    def resample_to_isotropic(self, image: sitk.Image) -> sitk.Image:

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [
            int(round(osz * ospc / tspc))
            for osz, ospc, tspc in zip(original_size, original_spacing, self.target_spacing)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(-1000)  # Air padding
        resample.SetInterpolator(sitk.sitkLinear)

        return resample.Execute(image)

    # ------------------------------------------------------------
    # 4. Clip + Normalize
    # ------------------------------------------------------------
    def clip_and_normalize(self, volume: np.ndarray) -> np.ndarray:

        volume = np.clip(volume, self.clip_min, self.clip_max)
        volume = (volume - self.clip_min) / (self.clip_max - self.clip_min)

        return volume.astype(np.float32)

    # ------------------------------------------------------------
    # 5. Extract Random 3D Patches
    # ------------------------------------------------------------
    def extract_random_patches(self, volume: np.ndarray) -> List[np.ndarray]:

        D, H, W = volume.shape
        pd, ph, pw = self.patch_size

        if D < pd or H < ph or W < pw:
            return []

        patches = []

        for _ in range(self.num_patches):

            d = np.random.randint(0, D - pd + 1)
            h = np.random.randint(0, H - ph + 1)
            w = np.random.randint(0, W - pw + 1)

            patch = volume[d:d+pd, h:h+ph, w:w+pw]
            patches.append(patch)

        return patches

    # ------------------------------------------------------------
    # 6. Full Patient Processing
    # ------------------------------------------------------------
    def preprocess_patient(
        self,
        patient_dir: str,
        output_dir: str,
        patient_id: str
    ) -> Dict:

        # 1. Load
        image = self.load_dicom_series(patient_dir)

        # 2. Resample
        image = self.resample_to_isotropic(image)

        # 3. Convert to HU
        volume = self.convert_to_hu(image)

        # 4. Clip & Normalize
        volume = self.clip_and_normalize(volume)

        # 5. Extract Patches
        patches = self.extract_random_patches(volume)

        if len(patches) == 0:
            raise ValueError(f"No valid patches for {patient_id}")

        # 6. Save
        patient_patch_dir = Path(output_dir) / "patches" / patient_id
        patient_patch_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for idx, patch in enumerate(patches):

            save_path = patient_patch_dir / f"patch_{idx}.npz"

            np.savez_compressed(
                save_path,
                patch=patch.astype(np.float16)
            )

            saved_paths.append(str(save_path))

        # Memory cleanup
        del volume
        gc.collect()

        return {
            "patient_id": patient_id,
            "num_patches": len(saved_paths),
            "patch_size": self.patch_size,
            "resampled_spacing": self.target_spacing,
            "patch_paths": saved_paths
        }


# ================================================================
# Batch Processing
# ================================================================

def batch_preprocess(
    raw_data_dir: str,
    output_dir: str,
    config: Dict,
    max_patients: Optional[int] = None
):

    raw_path = Path(raw_data_dir)

    patient_dirs = [
        folder for folder in raw_path.iterdir()
        if folder.is_dir() and len(list(folder.glob("*.dcm"))) > 50
    ]

    patient_dirs = sorted(patient_dirs)

    if max_patients:
        patient_dirs = patient_dirs[:max_patients]

    preprocessor = CTPreprocessor(
        clip_min=config.get("data.clip_min", -1000),
        clip_max=config.get("data.clip_max", 400),
        target_spacing=tuple(config.get("data.spacing", [1.0, 1.0, 1.0])),
        patch_size=tuple(config.get("data.input_shape", [1, 96, 96, 96])[1:]),
        num_patches=config.get("data.num_patches", 3)
    )

    metadata_list = []

    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        try:
            metadata = preprocessor.preprocess_patient(
                str(patient_dir),
                output_dir,
                patient_dir.name
            )
            metadata_list.append(metadata)

        except Exception as e:
            print(f"Error processing {patient_dir.name}: {str(e)}")
            continue

    metadata_df = pd.DataFrame(metadata_list)

    metadata_path = Path(output_dir) / "metadata" / "preprocessing_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df.to_csv(metadata_path, index=False)

    print(f"\nProcessed {len(metadata_list)} patients")
    print(f"Metadata saved to {metadata_path}")
