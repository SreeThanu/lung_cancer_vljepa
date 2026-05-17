"""
CT-JEPA Preprocessing Pipeline — Tolerant Version

Outputs:
  1. Full resampled CT volumes (.npz) for JEPA pretraining — all patients
  2. 96x96x96 nodule patches (.npz) with soft labels — annotated patients

Design principles:
  - Never crash on single patient failure; log errors to skipped_patients.csv
  - Accept nodules annotated by >= min_radiologists (not all 4)
  - Soft labels preserve score-3 uncertainty (mapped to 0.5)
  - Correct coordinate convention: xCoord/yCoord from XML are PIXEL INDICES,
    imageZposition is WORLD MM — convert px→world before voxel lookup
  - Parse XML manually; no pylidc dependency
"""

import gc
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


# ================================================================
# COORDINATE UTILITIES
# ================================================================

def world_to_voxel(
    world_coord: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Convert world mm coords (z, y, x) to voxel indices.
    origin/spacing are in ITK convention: (x, y, z).
    """
    x = int(round((world_coord[2] - origin[0]) / spacing[0]))
    y = int(round((world_coord[1] - origin[1]) / spacing[1]))
    z = int(round((world_coord[0] - origin[2]) / spacing[2]))
    return (z, y, x)


# ================================================================
# SERIES → PATIENT MAPPING
# ================================================================

def build_series_to_patient(raw_dir: str) -> Dict[str, str]:
    """
    Scan DICOM series folders and build series-UID → patient-ID mapping.
    Skips folders with no .dcm files or unreadable headers.
    """
    mapping: Dict[str, str] = {}
    raw_path = Path(raw_dir)

    dirs = [d for d in raw_path.iterdir() if d.is_dir()]

    for series_dir in tqdm(dirs, desc="Building series→patient map"):
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            patient_id = str(ds.PatientID).strip()
            mapping[series_dir.name] = patient_id
        except Exception:
            pass

    return mapping


# ================================================================
# XML ANNOTATION PARSER
# ================================================================

def _extract_patient_id(
    xml_file: Path,
    series_to_patient: Dict[str, str],
) -> Optional[str]:
    """Try path regex first, then fall back to XML tag parsing."""
    match = re.search(r"LIDC-IDRI-(\d{4})", str(xml_file))
    if match:
        return f"LIDC-IDRI-{match.group(1)}"

    try:
        tree = ET.parse(str(xml_file))
        root = tree.getroot()
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()
            if tag == "patientid" and elem.text:
                pid = elem.text.strip()
                if re.match(r"LIDC-IDRI-\d{4}", pid):
                    return pid
                if re.match(r"^\d{4}$", pid):
                    return f"LIDC-IDRI-{pid}"
            if tag in ("studyinstanceuid", "seriesinstanceuid") and elem.text:
                uid = elem.text.strip()
                if uid in series_to_patient:
                    return series_to_patient[uid]
    except Exception:
        pass

    return None


def _parse_xml_nodules(xml_path: str) -> List[Dict]:
    """
    Parse one LIDC-IDRI XML file.

    Returns flat list of nodule dicts, one per unblindedReadNodule element.
    Nodule IDs are NOT coordinated across radiologists — do NOT merge by ID here.

    Coordinate convention (CRITICAL):
      imageZposition → world mm  (z axis)
      xCoord, yCoord → PIXEL INDICES in DICOM image plane, NOT world mm
      Conversion to world mm happens later in preprocess_patient().
    """
    nodules: List[Dict] = []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return nodules

    series_uid: Optional[str] = None
    for elem in root.iter():
        if elem.tag.split("}")[-1].lower() == "seriesinstanceuid" and elem.text:
            series_uid = elem.text.strip()
            break

    for session in root.iter():
        if session.tag.split("}")[-1] != "readingSession":
            continue

        for nodule_elem in session:
            if nodule_elem.tag.split("}")[-1] != "unblindedReadNodule":
                continue

            nodule_id: Optional[str] = None
            scores: List[float] = []
            centroid_samples: List[Tuple[float, float, float]] = []

            for child in nodule_elem:
                tag = child.tag.split("}")[-1]
                if tag == "noduleID" and child.text:
                    nodule_id = child.text.strip()
                elif tag == "characteristics":
                    for c in child:
                        if "malignancy" in c.tag.lower() and c.text:
                            try:
                                scores.append(float(c.text))
                            except (ValueError, TypeError):
                                pass

            if nodule_id is None:
                continue

            # Extract per-slice centroids from ROI elements
            for roi in nodule_elem.iter():
                if roi.tag.split("}")[-1] != "roi":
                    continue

                z: Optional[float] = None
                xs: List[float] = []
                ys: List[float] = []
                locus_pts: List[Tuple[float, float]] = []

                for elem in roi:
                    t = elem.tag.split("}")[-1]

                    if t == "imageZposition":
                        try:
                            z = float(elem.text)
                        except (ValueError, TypeError):
                            pass

                    elif t == "edgeMap":
                        ex = ey = None
                        for pt in elem:
                            pt_tag = pt.tag.split("}")[-1]
                            if pt_tag == "xCoord":
                                try:
                                    ex = float(pt.text)
                                except (ValueError, TypeError):
                                    pass
                            elif pt_tag == "yCoord":
                                try:
                                    ey = float(pt.text)
                                except (ValueError, TypeError):
                                    pass
                        if ex is not None and ey is not None:
                            xs.append(ex)
                            ys.append(ey)

                    elif t == "locus":
                        # Fallback for small nodules marked with a point not a contour
                        lx = ly = None
                        for pt in elem:
                            pt_tag = pt.tag.split("}")[-1]
                            if pt_tag == "xCoord":
                                try:
                                    lx = float(pt.text)
                                except (ValueError, TypeError):
                                    pass
                            elif pt_tag == "yCoord":
                                try:
                                    ly = float(pt.text)
                                except (ValueError, TypeError):
                                    pass
                        if lx is not None and ly is not None:
                            locus_pts.append((lx, ly))

                # Prefer edgeMap contour centroid; fall back to locus point
                eff_xs = xs if xs else [p[0] for p in locus_pts]
                eff_ys = ys if ys else [p[1] for p in locus_pts]

                if z is not None and eff_xs:
                    centroid_samples.append((
                        z,                          # z: world mm
                        float(np.mean(eff_ys)),     # y: pixel index
                        float(np.mean(eff_xs)),     # x: pixel index
                    ))

            nodules.append({
                "nodule_id": nodule_id,
                "scores": scores,
                "centroid_samples": centroid_samples,
                "series_uid": series_uid,
            })

    return nodules


def parse_all_xml(
    xml_dir: str,
    series_to_patient: Dict[str, str],
) -> Dict:
    """
    Parse all XML files found recursively under xml_dir.

    Returns: patient_id → List[nodule_dict]
    Each element is one radiologist's annotation of one nodule.
    Spatial grouping (same physical nodule) happens in aggregate_nodules.
    """
    xml_files = sorted(Path(xml_dir).rglob("*.xml"))

    if not xml_files:
        print(f"[WARN] No XML files found in {xml_dir}")
        print("       Download LIDC-IDRI annotation XML package from TCIA and")
        print("       set preprocessing.xml_dir in jepa_config.yaml.")

    index: Dict = defaultdict(list)

    for xml_file in tqdm(xml_files, desc="Parsing XML annotations"):
        try:
            patient_id = _extract_patient_id(xml_file, series_to_patient)
            if patient_id is None:
                continue

            nodules = _parse_xml_nodules(str(xml_file))
            index[patient_id].extend(nodules)

        except Exception:
            pass

    return dict(index)


# ================================================================
# NODULE AGGREGATION WITH SOFT LABELS
# ================================================================

def _centroid_distance_mm(
    c1: Tuple[float, float, float],
    c2: Tuple[float, float, float],
    approx_in_plane_mm: float = 0.7,
) -> float:
    """Approximate 3D distance in mm. c = (z_mm, y_px, x_px)."""
    dz = c1[0] - c2[0]
    dy = (c1[1] - c2[1]) * approx_in_plane_mm
    dx = (c1[2] - c2[2]) * approx_in_plane_mm
    return float(np.sqrt(dz**2 + dy**2 + dx**2))


def aggregate_nodules(
    raw_index: Dict,
    min_radiologists: int,
    soft_label_map: Dict[int, float],
    spatial_threshold_mm: float = 10.0,
) -> Dict:
    """
    Group per-radiologist nodule annotations by spatial proximity, then
    aggregate each group into a single nodule record.

    Grouping: nodules whose centroids are within spatial_threshold_mm
    (single-linkage) are treated as the same physical nodule.
    A group needs >= min_radiologists total score entries to pass.

    raw_index: patient_id → List[nodule_dict]  (one dict per radiologist annotation)
    Returns:   patient_id → List[nodule_dict]  (one dict per physical nodule)
    """
    result: Dict = defaultdict(list)

    for patient_id, all_nodules in raw_index.items():
        # Compute centroid for each annotated nodule; skip those without coords
        valid: List[Dict] = []
        for nd in all_nodules:
            samples = nd.get("centroid_samples", [])
            if not samples:
                continue
            arr = np.array(samples, dtype=np.float64)
            centroid = (
                float(np.mean(arr[:, 0])),  # z: world mm
                float(np.mean(arr[:, 1])),  # y: pixel index
                float(np.mean(arr[:, 2])),  # x: pixel index
            )
            valid.append({**nd, "_centroid": centroid})

        if not valid:
            continue

        # Single-linkage spatial clustering
        assigned = [False] * len(valid)
        groups: List[List[int]] = []

        for i in range(len(valid)):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(valid)):
                    if assigned[j]:
                        continue
                    if any(
                        _centroid_distance_mm(valid[j]["_centroid"], valid[k]["_centroid"])
                        <= spatial_threshold_mm
                        for k in group
                    ):
                        group.append(j)
                        assigned[j] = True
                        changed = True
            groups.append(group)

        for group_idx, group in enumerate(groups):
            members = [valid[k] for k in group]
            all_scores = [float(s) for nd in members for s in nd["scores"]]

            if len(all_scores) < min_radiologists:
                continue

            avg_score = float(np.mean(all_scores))

            # Linear interpolation between nearest integer score keys
            lower = max(1, min(5, int(avg_score)))
            upper = max(1, min(5, lower + 1))
            frac = avg_score - lower

            if lower == upper or upper not in soft_label_map:
                soft_label = soft_label_map.get(lower, 0.5)
            else:
                soft_label = (
                    (1.0 - frac) * soft_label_map[lower]
                    + frac * soft_label_map[upper]
                )

            all_centroids = np.array([nd["_centroid"] for nd in members], dtype=np.float64)
            centroid = (
                float(np.mean(all_centroids[:, 0])),  # z: world mm
                float(np.mean(all_centroids[:, 1])),  # y: pixel index
                float(np.mean(all_centroids[:, 2])),  # x: pixel index
            )

            member_ids = sorted({nd["nodule_id"] for nd in members if nd.get("nodule_id")})
            composite_id = "_".join(member_ids) if member_ids else f"group_{group_idx}"

            series_uid = next(
                (nd["series_uid"] for nd in members if nd.get("series_uid")), None
            )

            result[patient_id].append({
                "nodule_id": composite_id,
                "scores": all_scores,
                "avg_score": avg_score,
                "soft_label": float(soft_label),
                "num_radiologists": len(all_scores),
                "centroid": centroid,
                "series_uid": series_uid,
            })

    return dict(result)


# ================================================================
# CT PREPROCESSOR
# ================================================================

class CTPreprocessor:
    def __init__(
        self,
        clip_min: float = -1000.0,
        clip_max: float = 400.0,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        patch_size: Tuple[int, int, int] = (96, 96, 96),
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target_spacing = target_spacing
        self.patch_size = patch_size

    def load_dicom_series(self, dicom_dir: str) -> sitk.Image:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_names:
            raise RuntimeError(f"No DICOM series found in {dicom_dir}")
        reader.SetFileNames(dicom_names)
        return reader.Execute()

    def resample(self, image: sitk.Image) -> sitk.Image:
        orig_spacing = image.GetSpacing()
        orig_size = image.GetSize()
        new_size = [
            int(round(sz * sp / tsp))
            for sz, sp, tsp in zip(orig_size, orig_spacing, self.target_spacing)
        ]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(self.target_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(-1000)
        return resample.Execute(image)

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        volume = np.clip(volume, self.clip_min, self.clip_max)
        volume = (volume - self.clip_min) / (self.clip_max - self.clip_min)
        return volume.astype(np.float32)

    def extract_patch(
        self,
        volume: np.ndarray,
        center: Tuple[int, int, int],
    ) -> np.ndarray:
        D, H, W = volume.shape
        pz, py, px = self.patch_size
        cz, cy, cx = center

        z0 = max(0, min(cz - pz // 2, D - pz))
        y0 = max(0, min(cy - py // 2, H - py))
        x0 = max(0, min(cx - px // 2, W - px))

        patch = volume[z0:z0+pz, y0:y0+py, x0:x0+px]

        if patch.shape != (pz, py, px):
            pad = [
                (0, pz - patch.shape[0]),
                (0, py - patch.shape[1]),
                (0, px - patch.shape[2]),
            ]
            patch = np.pad(patch, pad, constant_values=0.0)

        return patch

    def preprocess_patient(
        self,
        patient_dir: str,
        output_dir: str,
        patient_id: str,
        annotations: List[Dict],
        save_full_volume: bool = True,
    ) -> Dict:
        """
        Process one patient's CT scan.

        Steps:
          1. Load DICOM series
          2. Resample to isotropic spacing
          3. Normalize HU to [0, 1]
          4. Save full volume for JEPA pretraining (if save_full_volume)
          5. Extract and save 96x96x96 patches for each annotated nodule

        Raises RuntimeError on DICOM load failure (caller catches and logs).
        Per-nodule errors are caught internally and excluded from output.
        """
        output_path = Path(output_dir)

        image = self.load_dicom_series(patient_dir)

        # Store original geometry BEFORE resampling — required for coordinate conversion.
        # XML xCoord/yCoord are pixel indices in the ORIGINAL DICOM image plane.
        origin_orig = image.GetOrigin()    # (x, y, z) mm
        spacing_orig = image.GetSpacing()  # (x, y, z) mm/pixel

        image_resampled = self.resample(image)
        del image

        new_spacing = image_resampled.GetSpacing()

        # sitk returns array in (Z, Y, X) order
        volume = sitk.GetArrayFromImage(image_resampled).astype(np.float32)
        volume = self.normalize(volume)

        full_vol_path: Optional[str] = None

        if save_full_volume:
            vol_dir = output_path / "full_volumes"
            vol_dir.mkdir(parents=True, exist_ok=True)
            full_vol_path = str(vol_dir / f"{patient_id}.npz")
            np.savez_compressed(
                full_vol_path,
                volume=volume.astype(np.float16),
                patient_id=patient_id,
                spacing=np.array(new_spacing),
            )

        nodule_records: List[Dict] = []
        patch_paths: List[str] = []

        if annotations:
            patch_dir = output_path / "patches" / patient_id
            patch_dir.mkdir(parents=True, exist_ok=True)

            for nodule in annotations:
                try:
                    centroid_raw = nodule["centroid"]  # (z_world_mm, y_px, x_px)

                    # Convert pixel indices → world mm using ORIGINAL geometry.
                    # xCoord/yCoord from XML are pixel indices, NOT world mm.
                    x_world = origin_orig[0] + centroid_raw[2] * spacing_orig[0]
                    y_world = origin_orig[1] + centroid_raw[1] * spacing_orig[1]
                    z_world = centroid_raw[0]  # already world mm

                    # World → voxel in original space
                    centroid_vox_orig = world_to_voxel(
                        (z_world, y_world, x_world), origin_orig, spacing_orig
                    )

                    # Scale voxel coords to resampled space
                    scale = (
                        spacing_orig[2] / new_spacing[2],
                        spacing_orig[1] / new_spacing[1],
                        spacing_orig[0] / new_spacing[0],
                    )
                    centroid_vox = (
                        int(centroid_vox_orig[0] * scale[0]),
                        int(centroid_vox_orig[1] * scale[1]),
                        int(centroid_vox_orig[2] * scale[2]),
                    )

                    patch = self.extract_patch(volume, centroid_vox)

                    patch_name = f"{nodule['nodule_id']}.npz"
                    patch_path = str(patch_dir / patch_name)

                    np.savez_compressed(
                        patch_path,
                        patch=patch.astype(np.float16),
                        soft_label=np.float32(nodule["soft_label"]),
                        avg_score=np.float32(nodule["avg_score"]),
                        nodule_id=nodule["nodule_id"],
                        centroid_vox=np.array(centroid_vox),
                    )

                    patch_paths.append(patch_path)
                    nodule_records.append({
                        "patient_id": patient_id,
                        "nodule_id": nodule["nodule_id"],
                        "soft_label": nodule["soft_label"],
                        "avg_score": nodule["avg_score"],
                        "num_radiologists": nodule["num_radiologists"],
                        "centroid_z_world": centroid_raw[0],
                        "centroid_y_px": centroid_raw[1],
                        "centroid_x_px": centroid_raw[2],
                        "centroid_vox_z": centroid_vox[0],
                        "centroid_vox_y": centroid_vox[1],
                        "centroid_vox_x": centroid_vox[2],
                        "patch_shape": str(patch.shape),
                        "patch_path": patch_path,
                    })

                except Exception as e:
                    nodule_records.append({
                        "patient_id": patient_id,
                        "nodule_id": nodule.get("nodule_id", "unknown"),
                        "error": str(e),
                        "patch_path": None,
                    })

        del volume
        gc.collect()

        return {
            "patient_id": patient_id,
            "full_volume_path": full_vol_path,
            "num_patches": len(patch_paths),
            "patch_paths": patch_paths,
            "nodule_records": nodule_records,
        }


# ================================================================
# MAIN PIPELINE
# ================================================================

def run_pipeline(
    config,
    test_mode: bool = False,
    test_n: int = 5,
    patient_ids: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full CT-JEPA preprocessing pipeline.

    Args:
        config:      Loaded Config object (or plain dict) from jepa_config.yaml
        test_mode:   If True, process only test_n patients (with annotations)
        test_n:      Number of patients in test mode
        patient_ids: Explicit list of patient IDs to process (None = all)

    Returns:
        (metadata_df, labels_df) — both saved to processed_dir as CSV
    """
    # Support both Config object and plain dict
    cfg = config.data if hasattr(config, "data") else config

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]

    pre_cfg = cfg.get("preprocessing") or {}
    min_radiologists = pre_cfg.get("min_radiologists", 2)
    save_full_volumes = pre_cfg.get("save_full_volumes", True)
    xml_dir = pre_cfg.get("xml_dir", raw_dir)

    labels_cfg = cfg.get("labels") or {}
    raw_label_map = labels_cfg.get("soft_label_map", {1: 0.0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1.0})
    soft_label_map = {int(k): float(v) for k, v in raw_label_map.items()}

    output_path = Path(processed_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Series → patient mapping
    print("Step 1: Building series→patient mapping from DICOM headers...")
    series_to_patient = build_series_to_patient(raw_dir)
    n_patients = len(set(series_to_patient.values()))
    print(f"  {len(series_to_patient)} series → {n_patients} unique patients")

    # Step 2: Parse XML annotations
    print(f"Step 2: Parsing XML annotations from {xml_dir} ...")
    raw_index = parse_all_xml(xml_dir, series_to_patient)
    print(f"  Annotations found for {len(raw_index)} patients")

    # Step 3: Aggregate with soft labels
    print(f"Step 3: Aggregating nodules (min_radiologists={min_radiologists})...")
    patient_annotations = aggregate_nodules(raw_index, min_radiologists, soft_label_map)
    total_nodules = sum(len(v) for v in patient_annotations.values())
    print(f"  {len(patient_annotations)} patients, {total_nodules} valid nodules")

    # Step 4: Find CT series directories (>= 10 DCM files = CT volume, not SR)
    raw_path = Path(raw_dir)
    all_series_dirs = [d for d in raw_path.iterdir() if d.is_dir()]
    ct_series_dirs = [d for d in all_series_dirs if len(list(d.glob("*.dcm"))) >= 10]

    # Apply patient_ids filter
    if patient_ids is not None:
        pid_set = set(patient_ids)
        ct_series_dirs = [d for d in ct_series_dirs if series_to_patient.get(d.name) in pid_set]

    # Test mode: pick test_n unique patients that have annotations
    if test_mode:
        annot_patients = set(patient_annotations.keys())
        seen: set = set()
        test_dirs: List = []
        for d in ct_series_dirs:
            pid = series_to_patient.get(d.name)
            if pid in annot_patients and pid not in seen:
                test_dirs.append(d)
                seen.add(pid)
            if len(seen) >= test_n:
                break
        ct_series_dirs = test_dirs
        print(f"TEST MODE: {len(ct_series_dirs)} series, {len(seen)} annotated patients")

    # Step 5: Preprocess patients
    preprocessor = CTPreprocessor(
        clip_min=cfg["data"]["clip_min"],
        clip_max=cfg["data"]["clip_max"],
        target_spacing=tuple(cfg["data"]["spacing"]),
        patch_size=tuple(cfg["data"]["input_shape"][1:]),
    )

    metadata_rows: List[Dict] = []
    label_rows: List[Dict] = []
    skipped_log: List[Dict] = []
    seen_patients: set = set()

    for series_dir in tqdm(ct_series_dirs, desc="Preprocessing patients"):
        series_uid = series_dir.name
        patient_id = series_to_patient.get(series_uid)

        if patient_id is None or patient_id in seen_patients:
            continue
        seen_patients.add(patient_id)

        annotations = patient_annotations.get(patient_id, [])

        try:
            result = preprocessor.preprocess_patient(
                patient_dir=str(series_dir),
                output_dir=str(output_path),
                patient_id=patient_id,
                annotations=annotations,
                save_full_volume=save_full_volumes,
            )

            metadata_rows.append({
                "patient_id": patient_id,
                "series_uid": series_uid,
                "num_annotations": len(annotations),
                "num_patches_saved": result["num_patches"],
                "full_volume_path": result["full_volume_path"],
            })

            good_records = [r for r in result["nodule_records"] if "error" not in r]
            label_rows.extend(good_records)

            # Log nodule-level errors without dropping the patient
            for r in result["nodule_records"]:
                if "error" in r:
                    skipped_log.append({
                        "patient_id": patient_id,
                        "nodule_id": r.get("nodule_id", "unknown"),
                        "reason": r["error"],
                        "level": "nodule",
                    })

        except Exception as e:
            skipped_log.append({
                "patient_id": patient_id,
                "series_uid": series_uid,
                "reason": str(e),
                "level": "patient",
            })

    # Step 6: Save outputs
    metadata_df = pd.DataFrame(metadata_rows)
    labels_df = pd.DataFrame(label_rows)
    skipped_cols = ["patient_id", "series_uid", "nodule_id", "reason", "level"]
    skipped_df = pd.DataFrame(skipped_log, columns=skipped_cols) if skipped_log else pd.DataFrame(columns=skipped_cols)

    metadata_df.to_csv(output_path / "metadata.csv", index=False)
    labels_df.to_csv(output_path / "labels.csv", index=False)
    skipped_df.to_csv(output_path / "skipped_patients.csv", index=False)

    vol_count = len(list((output_path / "full_volumes").glob("*.npz"))) if save_full_volumes else 0

    print(f"\n{'='*42}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*42}")
    print(f"Patients processed:    {len(metadata_rows)}")
    print(f"Patients/nodules skipped: {len(skipped_log)}")
    print(f"Nodule patches saved:  {len(labels_df)}")
    if save_full_volumes:
        print(f"Full volumes saved:    {vol_count}")
    print(f"Outputs → {output_path}")
    print(f"{'='*42}")

    return metadata_df, labels_df
