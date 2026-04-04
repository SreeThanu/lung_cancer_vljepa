"""
Data exploration utilities for LIDC-IDRI dataset.

LIDC-IDRI Dataset Structure:
- DICOM-based CT scans (512x512 slices, variable number of slices)
- XML annotations from up to 4 radiologists per scan
- Nodule characteristics: size, malignancy (1-5 scale), spiculation, etc.
- Patient-level organization (LIDC-IDRI-XXXX)
"""

import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


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
            slice_thickness = abs(
                slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
            )

        return (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))

    def plot_slices(
        self,
        volume: np.ndarray,
        num_slices: int = 9,
        title: str = "CT Slices",
        cmap: str = "gray",
        figsize: Tuple[int, int] = (15, 10),
    ):
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
            axes[idx].set_title(f"Slice {slice_idx}/{depth}")
            axes[idx].axis("off")

        # Hide unused subplots
        for idx in range(len(indices), len(axes)):
            axes[idx].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_histogram(
        self,
        volume: np.ndarray,
        bins: int = 80,
        title: str = "HU Distribution",
        figsize: Tuple[int, int] = (12, 5),
    ):
        """
        Plot histogram of HU values.

        Args:
            volume: 3D volume in HU units
            bins: Number of histogram bins
            title: Plot title
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.hist(
            volume.flatten(),
            bins=bins,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
        )
        plt.xlabel("Hounsfield Units (HU)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14)
        plt.axvline(x=-1000, color="red", linestyle="--", label="Lung window min")
        plt.axvline(x=400, color="red", linestyle="--", label="Lung window max")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @classmethod
    def _iter_by_name(cls, element: ET.Element, name: str):
        for item in element.iter():
            if cls._local_name(item.tag) == name:
                yield item

    @classmethod
    def _find_first_text(cls, element: ET.Element, name: str) -> Optional[str]:
        for item in cls._iter_by_name(element, name):
            if item.text is not None:
                value = item.text.strip()
                if value:
                    return value
        return None

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_float(value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_patient_id(cls, root: ET.Element, xml_file: Path) -> Optional[str]:
        patient_tag_candidates = {
            "PatientId",
            "PatientID",
            "patientId",
            "patientID",
            "patient_id",
        }

        for item in root.iter():
            local_name = cls._local_name(item.tag)
            if local_name in patient_tag_candidates and item.text:
                patient_id = item.text.strip()
                if patient_id:
                    return patient_id

        path_match = re.search(r"LIDC-IDRI-\d{4}", str(xml_file))
        if path_match:
            return path_match.group(0)

        return None

    @classmethod
    def _matches_patient(
        cls,
        target_patient_id: str,
        xml_patient_id: Optional[str],
        series_uid: Optional[str],
        xml_file: Path,
    ) -> bool:
        target = target_patient_id.strip().upper()
        candidates = [xml_patient_id, series_uid, str(xml_file)]

        for candidate in candidates:
            if not candidate:
                continue
            value = str(candidate).upper()
            if target == value or target in value or value in target:
                return True

        return False

    @classmethod
    def _extract_centroid_from_nodule(
        cls, nodule_elem: ET.Element
    ) -> Optional[Tuple[float, float, float]]:
        x_coords: List[float] = []
        y_coords: List[float] = []
        z_coords: List[float] = []

        for roi in cls._iter_by_name(nodule_elem, "roi"):
            z_pos = cls._safe_float(cls._find_first_text(roi, "imageZposition"))
            if z_pos is not None:
                z_coords.append(z_pos)

            for edge_map in cls._iter_by_name(roi, "edgeMap"):
                x_coord = cls._safe_float(cls._find_first_text(edge_map, "xCoord"))
                y_coord = cls._safe_float(cls._find_first_text(edge_map, "yCoord"))

                if x_coord is not None:
                    x_coords.append(x_coord)
                if y_coord is not None:
                    y_coords.append(y_coord)

        if not x_coords or not y_coords or not z_coords:
            return None

        return (
            float(np.mean(z_coords)),
            float(np.mean(y_coords)),
            float(np.mean(x_coords)),
        )

    @staticmethod
    def malignancy_to_label(malignancy_mean: Optional[float]) -> Optional[int]:
        """
        Convert per-nodule average malignancy score to binary label.

        Rules:
        - >= 4.0: malignant (1)
        - <= 2.0: benign (0)
        - between 2 and 4: ambiguous (None, exclude)
        """
        if malignancy_mean is None:
            return None
        if malignancy_mean >= 4.0:
            return 1
        if malignancy_mean <= 2.0:
            return 0
        return None

    @staticmethod
    def label_to_name(label: Optional[int]) -> str:
        if label == 1:
            return "malignant"
        if label == 0:
            return "benign"
        return "ambiguous"

    def parse_xml_annotations(
        self, patient_id: Optional[str] = None, xml_root: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse XML annotation files and aggregate data per nodule.

        Extracted fields per aggregated nodule:
        - nodule_id
        - malignancy_scores (all radiologists)
        - malignancy_mean (average per nodule)
        - centroid coordinates (z, y, x) computed from ROI points
        - binary label using threshold >=4 malignant, <=2 benign, otherwise ambiguous

        Args:
            patient_id: Optional patient identifier filter.
            xml_root: Optional XML root directory. If None, searches under data_root.

        Returns:
            List of per-nodule annotation dictionaries.
        """
        if xml_root is None:
            search_root = self.data_root / patient_id if patient_id else self.data_root
        else:
            search_root = Path(xml_root)

        if not search_root.exists():
            return []

        xml_files = sorted(search_root.rglob("*.xml"))
        aggregated: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for xml_file in xml_files:
            try:
                root = ET.parse(xml_file).getroot()
            except ET.ParseError:
                continue

            xml_patient_id = self._extract_patient_id(root, xml_file)
            series_uid = self._find_first_text(root, "SeriesInstanceUid")

            if patient_id and not self._matches_patient(
                patient_id, xml_patient_id, series_uid, xml_file
            ):
                continue

            current_patient_id = xml_patient_id or patient_id or "unknown"

            for session_idx, session in enumerate(self._iter_by_name(root, "readingSession")):
                for nodule_idx, nodule_elem in enumerate(
                    self._iter_by_name(session, "unblindedReadNodule")
                ):
                    raw_nodule_id = self._find_first_text(nodule_elem, "noduleID")
                    nodule_id = (
                        raw_nodule_id
                        if raw_nodule_id
                        else f"session_{session_idx}_nodule_{nodule_idx}"
                    )

                    malignancy = self._safe_int(
                        self._find_first_text(nodule_elem, "malignancy")
                    )
                    centroid = self._extract_centroid_from_nodule(nodule_elem)

                    key = (current_patient_id, series_uid or xml_file.stem, nodule_id)
                    if key not in aggregated:
                        aggregated[key] = {
                            "patient_id": current_patient_id,
                            "series_uid": series_uid,
                            "nodule_id": nodule_id,
                            "malignancy_scores": [],
                            "centroid_samples": [],
                            "source_xml": str(xml_file),
                        }

                    if malignancy is not None:
                        aggregated[key]["malignancy_scores"].append(malignancy)

                    if centroid is not None:
                        aggregated[key]["centroid_samples"].append(centroid)

        results: List[Dict[str, Any]] = []

        for _, aggregated_row in aggregated.items():
            malignancy_scores = aggregated_row["malignancy_scores"]
            centroid_samples = aggregated_row["centroid_samples"]

            malignancy_mean = (
                float(np.mean(malignancy_scores)) if len(malignancy_scores) > 0 else None
            )

            if len(centroid_samples) > 0:
                centroid_array = np.array(centroid_samples, dtype=float)
                centroid_z = float(np.mean(centroid_array[:, 0]))
                centroid_y = float(np.mean(centroid_array[:, 1]))
                centroid_x = float(np.mean(centroid_array[:, 2]))
            else:
                centroid_z = None
                centroid_y = None
                centroid_x = None

            label = self.malignancy_to_label(malignancy_mean)

            results.append(
                {
                    "patient_id": aggregated_row["patient_id"],
                    "series_uid": aggregated_row["series_uid"],
                    "nodule_id": aggregated_row["nodule_id"],
                    "source_xml": aggregated_row["source_xml"],
                    "malignancy_scores": malignancy_scores,
                    "malignancy_mean": malignancy_mean,
                    "malignancy": malignancy_mean,
                    "label": label,
                    "label_name": self.label_to_name(label),
                    "is_ambiguous": label is None,
                    "centroid_z": centroid_z,
                    "centroid_y": centroid_y,
                    "centroid_x": centroid_x,
                    "radiologist_count": len(malignancy_scores),
                }
            )

        results.sort(
            key=lambda row: (
                row.get("patient_id") or "",
                row.get("nodule_id") or "",
                row.get("source_xml") or "",
            )
        )
        return results

    def get_patient_label(self, annotations: List[Dict[str, Any]]) -> Optional[int]:
        """
        Assign patient-level label using per-nodule labels.

        Rule: patient is malignant if ANY nodule is malignant.
        """
        nodule_labels = [
            item.get("label") for item in annotations if item.get("label") in (0, 1)
        ]

        if any(label == 1 for label in nodule_labels):
            return 1
        if any(label == 0 for label in nodule_labels):
            return 0
        return None

    def compute_patient_labels(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, Optional[int]]:
        """Compute patient-level labels from parsed nodule annotations."""
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for item in annotations:
            patient_id = item.get("patient_id") or "unknown"
            grouped[patient_id].append(item)

        return {
            patient_id: self.get_patient_label(items)
            for patient_id, items in grouped.items()
        }

    def get_label_distribution(self, annotations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count benign/ambiguous/malignant nodules from parsed annotations."""
        distribution = {"benign": 0, "ambiguous": 0, "malignant": 0}

        for item in annotations:
            label_name = item.get("label_name", self.label_to_name(item.get("label")))
            if label_name not in distribution:
                distribution[label_name] = 0
            distribution[label_name] += 1

        return distribution

    def create_summary_df(self, annotations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create summary DataFrame from annotations.

        Args:
            annotations: List of nodule annotations

        Returns:
            Pandas DataFrame with summary statistics
        """
        if not annotations:
            return pd.DataFrame()

        # New aggregated nodule structure
        if "malignancy_scores" in annotations[0]:
            rows = []
            for ann in annotations:
                rows.append(
                    {
                        "patient_id": ann.get("patient_id"),
                        "series_uid": ann.get("series_uid"),
                        "nodule_id": ann.get("nodule_id"),
                        "malignancy_scores": ann.get("malignancy_scores", []),
                        "malignancy_mean": ann.get("malignancy_mean"),
                        "label": ann.get("label"),
                        "label_name": ann.get("label_name"),
                        "centroid_z": ann.get("centroid_z"),
                        "centroid_y": ann.get("centroid_y"),
                        "centroid_x": ann.get("centroid_x"),
                        "radiologist_count": ann.get("radiologist_count", 0),
                    }
                )
            return pd.DataFrame(rows)

        # Backward-compatible fallback for older annotation format
        data = []
        for ann in annotations:
            chars = ann.get("characteristics", {})
            data.append(
                {
                    "nodule_id": ann.get("nodule_id"),
                    "malignancy": chars.get("malignancy", None),
                    "subtlety": chars.get("subtlety", None),
                    "texture": chars.get("texture", None),
                    "sphericity": chars.get("sphericity", None),
                    "margin": chars.get("margin", None),
                }
            )

        return pd.DataFrame(data)


def visualize_preprocessing_comparison(
    original: np.ndarray, processed: np.ndarray, slice_idx: Optional[int] = None
):
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

    axes[0].imshow(original[slice_idx], cmap="gray")
    axes[0].set_title(f"Original (Slice {slice_idx})")
    axes[0].axis("off")

    axes[1].imshow(processed[slice_idx], cmap="gray")
    axes[1].set_title("Preprocessed (Normalized)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
