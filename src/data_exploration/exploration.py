"""
Data exploration utilities for LIDC-IDRI dataset.

LIDC-IDRI Dataset Structure:
- DICOM-based CT scans (512x512 slices, variable number of slices)
- XML annotations from up to 4 radiologists per scan
- Nodule characteristics: size, malignancy (1-5 scale), spiculation, etc.
- Patient-level organization (LIDC-IDRI-XXXX)
"""

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import xml.etree.ElementTree as ET


class LIDCExplorer:
    """Exploration utilities for LIDC-IDRI dataset."""
    
    def __init__(self, data_root: str):
        """
        Initialize explorer.
        
        Args:
            data_root: Path to LIDC-IDRI root directory
        """
        self.data_root = Path(data_root)
        
    def list_patients(self) -> List[str]:
        """
        Return only Series folders that likely contain full CT volumes.
        Filters out tiny series (e.g., localizers).
        """
    
        patients = []
    
        for folder in self.data_root.iterdir():
            if folder.is_dir():
                dcm_files = list(folder.glob("*.dcm"))
    
                # Keep only folders with sufficient slices
                if len(dcm_files) > 50:
                    patients.append(folder.name)
    
        return sorted(patients)


    def load_dicom_scan(self, patient_id: str) -> List[pydicom.FileDataset]:
        """
        Load CT DICOM slices for a patient/study.
    
        Automatically:
        - Recursively scans all DICOM files
        - Filters only CT modality images
        - Groups by SeriesInstanceUID
        - Selects the largest CT series
        - Sorts slices by Z coordinate
    
        Args:
            patient_id: StudyInstanceUID or patient folder
    
        Returns:
            List of sorted CT slices
        """
    
        patient_path = self.data_root / patient_id
    
        if not patient_path.exists():
            raise FileNotFoundError(f"Patient directory not found: {patient_path}")
    
        # Collect all DICOM files recursively
        dcm_files = []
        for root, _, files in os.walk(patient_path):
            for file in files:
                if file.lower().endswith(".dcm"):
                    dcm_files.append(os.path.join(root, file))
    
        if not dcm_files:
            raise FileNotFoundError(f"No DICOM files found in {patient_id}")
    
        print(f"Found {len(dcm_files)} total DICOM files for {patient_id}")
    
        # Group CT slices by SeriesInstanceUID
        series_dict = {}
    
        for file_path in dcm_files:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=False, force=True)
    
                # Keep only CT images with spatial position
                if (
                    getattr(ds, "Modality", None) == "CT"
                    and hasattr(ds, "ImagePositionPatient")
                    and hasattr(ds, "SeriesInstanceUID")
                ):
                    series_uid = ds.SeriesInstanceUID
    
                    if series_uid not in series_dict:
                        series_dict[series_uid] = []
    
                    series_dict[series_uid].append(ds)
    
            except Exception:
                continue  # Skip corrupted or unreadable files
    
        if not series_dict:
            raise ValueError(f"No valid CT slices found for {patient_id}")
    
        # Select the series with the most slices
        selected_series_uid = max(series_dict, key=lambda k: len(series_dict[k]))
        slices = series_dict[selected_series_uid]
    
        print(f"Selected CT series: {selected_series_uid}")
        print(f"Number of CT slices: {len(slices)}")
    
        # Sort slices by Z coordinate
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
        return slices


    def get_pixels_hu(self, slices: List[pydicom.FileDataset]) -> np.ndarray:
        """
        Convert DICOM pixel data to Hounsfield Units (HU).
        
        HU Formula: HU = pixel * slope + intercept
        HU Scale: Air ≈ -1000, Water = 0, Soft tissue = 40-80, Bone > 400
        
        Args:
            slices: List of DICOM slice objects
            
        Returns:
            3D numpy array in HU units [D, H, W]
        """
        # Stack pixel arrays
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)

        # Handle outside-scanner pixels
        image[image == -2000] = 0
        
        # Convert to HU using rescale slope and intercept
        for slice_idx in range(len(slices)):
            intercept = slices[slice_idx].RescaleIntercept
            slope = slices[slice_idx].RescaleSlope
            
            if slope != 1:
                image[slice_idx] = slope * image[slice_idx].astype(np.float64)
                image[slice_idx] = image[slice_idx].astype(np.int16)
                
            image[slice_idx] += np.int16(intercept)
            
        return np.array(image, dtype=np.int16)

    def get_spacing(self, slices: List[pydicom.FileDataset]) -> Tuple[float, float, float]:
        """
        Extract voxel spacing (mm) from DICOM metadata.
        
        Args:
            slices: List of DICOM slices
            
        Returns:
            Tuple of (z_spacing, y_spacing, x_spacing) in mm
        """
        # In-plane pixel spacing (x, y)
        pixel_spacing = slices[0].PixelSpacing
        
        # Slice thickness (z)
        try:
            slice_thickness = slices[0].SliceThickness
        except AttributeError:
            # Calculate from slice positions
            slice_thickness = abs(slices[1].ImagePositionPatient[2] - 
                                slices[0].ImagePositionPatient[2])
        
        return (float(slice_thickness), 
                float(pixel_spacing[0]), 
                float(pixel_spacing[1]))

    def plot_slices(self, 
                    volume: np.ndarray, 
                    num_slices: int = 9,
                    title: str = "CT Slices",
                    cmap: str = 'gray',
                    figsize: Tuple[int, int] = (15, 10)):
        """
        Plot multiple slices from 3D volume.
        
        Args:
            volume: 3D numpy array [D, H, W]
            num_slices: Number of slices to display
            title: Plot title
            cmap: Colormap
            figsize: Figure size
        """
        depth = volume.shape[0]
        indices = np.linspace(0, depth - 1, num_slices, dtype=int)
        
        rows = int(np.sqrt(num_slices))
        cols = int(np.ceil(num_slices / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, slice_idx in enumerate(indices):
            axes[idx].imshow(volume[slice_idx], cmap=cmap)
            axes[idx].set_title(f'Slice {slice_idx}/{depth}')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(indices), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, 
                       volume: np.ndarray, 
                       bins: int = 80,
                       title: str = "HU Distribution",
                       figsize: Tuple[int, int] = (12, 5)):
        """
        Plot histogram of HU values.
        
        Args:
            volume: 3D volume in HU units
            bins: Number of histogram bins
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.hist(volume.flatten(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Hounsfield Units (HU)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14)
        plt.axvline(x=-1000, color='red', linestyle='--', label='Lung window min')
        plt.axvline(x=400, color='red', linestyle='--', label='Lung window max')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def parse_xml_annotations(self, patient_id: str) -> List[Dict]:
        """
        Parse XML annotation files for nodule information.
        
        XML Structure:
        - Multiple <readingSession> elements (one per radiologist)
        - Each contains <unblindedReadNodule> with:
            - Nodule characteristics (malignancy, spiculation, etc.)
            - ROI coordinates for each slice
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            List of nodule annotations
        """
        patient_path = self.data_root / patient_id
        xml_files = list(patient_path.rglob("*.xml"))
        
        nodules = []
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for session in root.findall('.//readingSession'):
                for nodule in session.findall('.//unblindedReadNodule'):
                    nodule_info = {
                        'nodule_id': nodule.find('noduleID').text if nodule.find('noduleID') is not None else None,
                        'characteristics': {}
                    }
                    
                    # Extract characteristics
                    chars = nodule.find('characteristics')
                    if chars is not None:
                        for char in chars:
                            nodule_info['characteristics'][char.tag] = char.text
                    
                    nodules.append(nodule_info)
        
        return nodules

    def create_summary_df(self, annotations: List[Dict]) -> pd.DataFrame:
        """
        Create summary DataFrame from annotations.
        
        Args:
            annotations: List of nodule annotations
            
        Returns:
            Pandas DataFrame with summary statistics
        """
        if not annotations:
            return pd.DataFrame()
        
        data = []
        for ann in annotations:
            chars = ann['characteristics']
            data.append({
                'nodule_id': ann['nodule_id'],
                'malignancy': chars.get('malignancy', None),
                'subtlety': chars.get('subtlety', None),
                'texture': chars.get('texture', None),
                'sphericity': chars.get('sphericity', None),
                'margin': chars.get('margin', None),
            })
        
        return pd.DataFrame(data)


def visualize_preprocessing_comparison(original: np.ndarray, 
                                       processed: np.ndarray,
                                       slice_idx: Optional[int] = None):
    """
    Compare original and preprocessed CT slices side-by-side.
    
    Args:
        original: Original CT volume
        processed: Preprocessed CT volume
        slice_idx: Slice index to display (middle slice if None)
    """
    if slice_idx is None:
        slice_idx = original.shape[0] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(original[slice_idx], cmap='gray')
    axes[0].set_title(f'Original (Slice {slice_idx})')
    axes[0].axis('off')
    
    axes[1].imshow(processed[slice_idx], cmap='gray')
    axes[1].set_title(f'Preprocessed (Normalized)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
