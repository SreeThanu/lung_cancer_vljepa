# ================================================================
# CT Scan Preprocessing Pipeline (FIXED VERSION)
# ================================================================

import gc
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ================================================================
# UTILITIES
# ================================================================

def world_to_voxel(world_coord, origin, spacing):
    """
    world_coord: (z, y, x) from XML
    origin: (x, y, z) from ITK
    spacing: (x, y, z)
    """

    x = int(round((world_coord[2] - origin[0]) / spacing[0]))
    y = int(round((world_coord[1] - origin[1]) / spacing[1]))
    z = int(round((world_coord[0] - origin[2]) / spacing[2]))

    return (z, y, x)

# ================================================================
# PREPROCESSOR
# ================================================================

class CTPreprocessor:
    def __init__(
        self,
        clip_min=-1000,
        clip_max=400,
        target_spacing=(1, 1, 1),
        patch_size=(96, 96, 96),
        num_patches=1,
        jitter=0,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.jitter = jitter

    # ------------------------------------------------------------
    def load_dicom_series(self, dicom_dir):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        reader.SetFileNames(dicom_names)
        return reader.Execute()

    # ------------------------------------------------------------
    def convert_to_hu(self, image):
        volume = sitk.GetArrayFromImage(image).astype(np.float32)

        intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0.0
        slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1.0

        return volume * slope + intercept

    # ------------------------------------------------------------
    def resample(self, image):
        spacing = image.GetSpacing()
        size = image.GetSize()

        new_size = [
            int(round(sz * sp / tsp))
            for sz, sp, tsp in zip(size, spacing, self.target_spacing)
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(-1000)

        return resample.Execute(image)

    # ------------------------------------------------------------
    def normalize(self, vol):
        vol = np.clip(vol, self.clip_min, self.clip_max)
        vol = (vol - self.clip_min) / (self.clip_max - self.clip_min)
        return vol.astype(np.float32)

    # ------------------------------------------------------------
    def extract_patch(self, volume, center):
        D, H, W = volume.shape
        pz, py, px = self.patch_size

        cz, cy, cx = center

        z0 = max(0, min(cz - pz // 2, D - pz))
        y0 = max(0, min(cy - py // 2, H - py))
        x0 = max(0, min(cx - px // 2, W - px))

        patch = volume[z0:z0+pz, y0:y0+py, x0:x0+px]

        if patch.shape != (pz, py, px):
            pad = [(0, pz - patch.shape[0]),
                   (0, py - patch.shape[1]),
                   (0, px - patch.shape[2])]
            patch = np.pad(patch, pad, constant_values=0)

        return patch

    # ------------------------------------------------------------
    def refine_centroid(self, volume, centroid_vox):

        patch = self.extract_patch(volume, centroid_vox)
    
        # 🔥 Step 1: focus on soft tissue range (nodules)
        mask = (patch > 0.4) & (patch < 0.85)
    
        coords = np.argwhere(mask)
    
        if len(coords) < 50:
            return centroid_vox  # fallback
    
        # 🔥 Step 2: cluster center
        center = coords.mean(axis=0).astype(int)
    
        refined = (
            centroid_vox[0] + (center[0] - patch.shape[0] // 2),
            centroid_vox[1] + (center[1] - patch.shape[1] // 2),
            centroid_vox[2] + (center[2] - patch.shape[2] // 2),
        )
    
        return refined

    # ------------------------------------------------------------
    def preprocess_patient(self, patient_dir, output_dir, patient_id, annotations):
        image = self.load_dicom_series(patient_dir)
    
        # 🔥 ORIGINAL geometry (before resampling)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
    
        # 🔥 RESAMPLE image
        image = self.resample(image)
    
        new_spacing = image.GetSpacing()
    
        volume = self.convert_to_hu(image)
        volume = self.normalize(volume)
    
        save_dir = Path(output_dir) / "patches" / patient_id
        save_dir.mkdir(parents=True, exist_ok=True)
    
        records = []
        paths = []
    
        for nodule in annotations:
    
            if nodule["label"] is None:
                continue
    
            centroid_raw = nodule["centroid"]  # (z_world_mm, y_pixel, x_pixel) — mixed format from LIDC XML
    
            # 🔥 FIX (Bug 1): Convert x/y from pixel indices → world mm using original DICOM geometry.
            # In LIDC-IDRI XML: imageZposition is already in world mm, but xCoord/yCoord are
            # pixel indices in the DICOM image — NOT world coordinates. Treating them as world mm
            # causes a centroid displacement of hundreds of pixels, placing patches in air/ribs.
            x_world = origin[0] + centroid_raw[2] * spacing[0]
            y_world = origin[1] + centroid_raw[1] * spacing[1]
            z_world = centroid_raw[0]  # already in world mm
            centroid_world = (z_world, y_world, x_world)
    
            # Step 1: world → voxel in original (pre-resample) space
            centroid_vox = world_to_voxel(centroid_world, origin, spacing)
    
            # Step 2: scale voxel coords to resampled space
            scale = (
                spacing[2] / new_spacing[2],
                spacing[1] / new_spacing[1],
                spacing[0] / new_spacing[0],
            )
    
            centroid_vox = (
                int(centroid_vox[0] * scale[0]),
                int(centroid_vox[1] * scale[1]),
                int(centroid_vox[2] * scale[2]),
            )
    
            # Step 3: refine centroid to soft-tissue cluster center
            centroid_vox = self.refine_centroid(volume, centroid_vox)
    
            for i in range(self.num_patches):
    
                patch = self.extract_patch(volume, centroid_vox)
    
                name = f"{nodule['nodule_id']}_{i}.npz"
                path = save_dir / name
    
                np.savez_compressed(
                    path,
                    patch=patch.astype(np.float16),
                    label=np.int8(nodule["label"]),
                    nodule_id=nodule["nodule_id"],
                    centroid_vox=np.array(centroid_vox),
                )
    
                paths.append(str(path))
    
                records.append({
                    "patient_id": patient_id,
                    "nodule_id": nodule["nodule_id"],
                    "label": nodule["label"],
                    "avg_malignancy_score": nodule["avg_score"],
                    "centroid_z": centroid_raw[0],
                    "centroid_y": centroid_raw[1],
                    "centroid_x": centroid_raw[2],
                    "patch_path": str(path),
                })
    
        if len(paths) == 0:
            raise ValueError("No valid nodules")
    
        del volume
        gc.collect()
    
        return {
            "patient_id": patient_id,
            "num_patches": len(paths),
            "patch_paths": paths,
            "nodule_records": records,
        }

# ================================================================
# BATCH PROCESS
# ================================================================

def batch_preprocess(raw_dir, output_dir, config, annotations):

    raw_path = Path(raw_dir)

    dirs = [d for d in raw_path.iterdir() if d.is_dir()]

    preprocessor = CTPreprocessor(
        clip_min=config["data"]["clip_min"],
        clip_max=config["data"]["clip_max"],
        target_spacing=tuple(config["data"]["spacing"]),
        patch_size=tuple(config["data"]["input_shape"][1:]),
        num_patches=config["data"]["num_patches"],
        jitter=config["data"]["jitter"],
    )

    meta = []
    labels = []

    for d in tqdm(dirs):

        sid = d.name
        patient_id = series_to_patient.get(sid)

        if patient_id is None:
            continue
        
        nodules = annotations.get(patient_id, [])

        if len(nodules) == 0:
            continue

        try:
            out = preprocessor.preprocess_patient(
                str(d), output_dir, patient_id, nodules
            )
            
            meta.append(out)
            labels.extend(out["nodule_records"])

        except Exception as e:
            print("Error:", e)

    return pd.DataFrame(meta), pd.DataFrame(labels)