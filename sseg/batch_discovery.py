from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import nrrd
import numpy as np

from .batch_models import BatchDiscoveryResult, BatchDiscoverySettings, DiscoveredExam, SeriesFileInfo
from .io_utils import determine_nrrd_space, reorient_affine_space_to_ras


def _normalize_extensions(exts: list[str]) -> list[str]:
    normalized = []
    for ext in exts:
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext.lower())
    return normalized or ['.nii', '.nii.gz', '.nrrd']


def strip_known_image_extension(filename: str) -> str:
    lower = filename.lower()
    for ext in ('.nii.gz', '.nii', '.nrrd'):
        if lower.endswith(ext):
            return filename[:-len(ext)]
    return Path(filename).stem


def is_supported_image_file(filename: str, image_extensions: list[str]) -> bool:
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in _normalize_extensions(image_extensions))


def is_obvious_nonseries_file(filename: str, settings: BatchDiscoverySettings) -> bool:
    lower = filename.lower()
    suffixes = list(settings.brainmask_suffixes) + list(settings.segmentation_suffixes)
    return any(lower.endswith(sfx.lower()) for sfx in suffixes if sfx)


def match_volume_suffix(filename: str, volume_suffixes: list[str]) -> Optional[str]:
    """Return the configured volume suffix that ends the filename (longest wins), else None."""
    lower = filename.lower()
    matches = [sfx for sfx in volume_suffixes if sfx and lower.endswith(sfx.lower())]
    return max(matches, key=len) if matches else None


def _infer_series_type_and_suffix(filename: str, volume_suffixes: list[str]) -> tuple[str, str]:
    """Split a filename into (series type, volume suffix).

    002_d-12_T1c_brain-norm.nii.gz with suffix '_brain-norm.nii.gz' -> ('T1c', '_brain-norm.nii.gz').
    Returns ('', '') when no configured suffix matches.
    """
    suffix = match_volume_suffix(filename, volume_suffixes)
    if suffix is None:
        return '', ''
    base = filename[:len(filename) - len(suffix)].rstrip('_')
    return base.split('_')[-1], suffix


def _round_float_list(values, ndigits=5):
    return [round(float(v), ndigits) for v in values]


def _nrrd_affine_from_header(header):
    affine = np.eye(4)
    if 'space directions' in header and 'space origin' in header:
        space_directions = np.array(header['space directions'])
        if space_directions.shape == (4, 3):
            affine[:3, :3] = space_directions[1:, :]
        else:
            affine[:3, :3] = space_directions
        affine[:3, 3] = np.array(header['space origin'])
    space = header.get('space', 'right-anterior-superior')
    affine = reorient_affine_space_to_ras(affine, space)
    return affine


def read_geometry_signature(file_path: str) -> tuple[tuple[int, ...], tuple[float, ...], str]:
    if file_path.lower().endswith('.nrrd'):
        header = nrrd.read_header(file_path)
        shape = tuple(int(x) for x in header.get('sizes', []))
        space_dirs = np.array(header.get('space directions', np.eye(3)))
        if space_dirs.ndim == 2 and space_dirs.shape[0] >= 3:
            if space_dirs.shape == (4, 3):
                sd = space_dirs[1:, :]
            else:
                sd = space_dirs[:3, :3]
            voxel_dims = tuple(float(np.linalg.norm(sd[i])) for i in range(3))
        else:
            voxel_dims = (1.0, 1.0, 1.0)
        affine = _nrrd_affine_from_header(header)
        affine_key = '|'.join(str(x) for x in _round_float_list(affine.flatten(), ndigits=4))
        return shape, voxel_dims, affine_key

    img = nib.load(file_path)
    ras_img = nib.as_closest_canonical(img)
    shape = tuple(int(x) for x in ras_img.shape[:3])
    voxel_dims = tuple(float(x) for x in ras_img.header.get_zooms()[:3])
    affine_key = '|'.join(str(x) for x in _round_float_list(ras_img.affine.flatten(), ndigits=4))
    return shape, voxel_dims, affine_key


def classify_directory_as_exam(path: Path, settings: BatchDiscoverySettings):
    if not path.is_dir():
        return False, 'not a directory', None
    parent = path.parent
    if parent is None or parent == path:
        return False, 'missing parent directory', None

    patient_id = parent.name
    exam_id = path.name
    if not patient_id or not exam_id.startswith(patient_id):
        return False, 'directory name does not start with parent patient id', None

    top_level_files = [p for p in path.iterdir() if p.is_file()]
    image_files = [p for p in top_level_files if is_supported_image_file(p.name, settings.image_extensions)]
    if not image_files:
        return False, 'no supported top-level image files', None

    excluded = [p for p in image_files if is_obvious_nonseries_file(p.name, settings)]
    series = [
        p for p in image_files
        if p not in excluded and p.name.startswith(patient_id)
        and match_volume_suffix(p.name, settings.volume_suffixes)
    ]
    if not series:
        return False, 'no image files ending in a configured volume suffix', None

    subdirs = sorted([p.name for p in path.iterdir() if p.is_dir()])
    exam = DiscoveredExam(
        patient_id=patient_id,
        exam_id=exam_id,
        patient_dir=str(parent),
        exam_dir=str(path),
        series_volume_paths=sorted(str(p) for p in series),
        excluded_nonseries_paths=sorted(str(p) for p in excluded),
        subdirs_present=subdirs,
    )
    analyze_exam(exam, settings)
    return True, 'astril-like exam directory', exam


def analyze_exam(exam: DiscoveredExam, settings: Optional[BatchDiscoverySettings] = None) -> DiscoveredExam:
    settings = settings or BatchDiscoverySettings(root_dir=exam.patient_dir)
    exam.series_files = []
    exam.available_suffixes = []
    exam.available_series_types_by_suffix = {}
    exam.compatible_groups = {}
    exam.preferred_group_by_suffix = {}

    groups: dict[str, list[str]] = defaultdict(list)
    suffix_group_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
    suffix_group_series: dict[tuple[str, str], set[str]] = defaultdict(set)
    suffix_series_all: dict[str, set[str]] = defaultdict(set)
    verify = bool(getattr(settings, 'verify_scan_compatibility', False))

    for path_str in sorted(exam.series_volume_paths):
        path = Path(path_str)
        series_type, shared_suffix = _infer_series_type_and_suffix(path.name, settings.volume_suffixes)
        suffix_series_all[shared_suffix].add(series_type)
        if verify:
            try:
                shape, voxel_dims, affine_key = read_geometry_signature(str(path))
            except Exception:
                continue
            group_key = f"shape={shape};zooms={tuple(round(v, 5) for v in voxel_dims)};affine={affine_key}"
            shape_list = list(shape)
            zoom_list = _round_float_list(voxel_dims, ndigits=5)
        else:
            group_key = f"unverified_suffix={shared_suffix}"
            shape_list = []
            zoom_list = []

        exam.series_files.append(SeriesFileInfo(
            path=str(path),
            filename=path.name,
            series_type=series_type,
            shared_suffix=shared_suffix,
            compatible_group_key=group_key,
            shape=shape_list,
            voxel_dims=zoom_list,
        ))
        groups[group_key].append(str(path))
        suffix_group_paths[(shared_suffix, group_key)].append(str(path))
        suffix_group_series[(shared_suffix, group_key)].add(series_type)

    exam.compatible_groups = {k: sorted(v) for k, v in groups.items()}

    suffixes = sorted({sf.shared_suffix for sf in exam.series_files if sf.shared_suffix})
    exam.available_suffixes = suffixes
    for suffix in suffixes:
        if verify:
            candidates = []
            for (sfx, gkey), paths in suffix_group_paths.items():
                if sfx != suffix:
                    continue
                candidates.append((len(paths), len(suffix_group_series[(sfx, gkey)]), gkey))
            if not candidates:
                continue
            candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
            chosen_group = candidates[0][2]
            exam.preferred_group_by_suffix[suffix] = chosen_group
            exam.available_series_types_by_suffix[suffix] = sorted(suffix_group_series[(suffix, chosen_group)])
        else:
            chosen_group = f"unverified_suffix={suffix}"
            exam.preferred_group_by_suffix[suffix] = chosen_group
            exam.available_series_types_by_suffix[suffix] = sorted(suffix_series_all.get(suffix, set()))

    # brainmask preference: exam-level first, then patient-level fallback
    exam.brainmask_path = None
    exam_dir = Path(exam.exam_dir)
    patient_dir = Path(exam.patient_dir)
    for sfx in settings.brainmask_suffixes:
        matches = sorted(p for p in exam_dir.iterdir() if p.is_file() and p.name.endswith(sfx))
        if matches:
            exam.brainmask_path = str(matches[0])
            break
    if exam.brainmask_path is None:
        for sfx in settings.brainmask_suffixes:
            matches = sorted(p for p in patient_dir.iterdir() if p.is_file() and p.name.endswith(sfx))
            if matches:
                exam.brainmask_path = str(matches[0])
                break

    # segmentation preference list in order
    seg_paths = []
    seen = set()
    for sfx in settings.segmentation_suffixes:
        for base_dir in (exam_dir, patient_dir):
            for p in sorted(base_dir.iterdir() if base_dir.exists() else []):
                if p.is_file() and p.name.endswith(sfx):
                    sp = str(p)
                    if sp not in seen:
                        seg_paths.append(sp)
                        seen.add(sp)
    exam.segmentation_paths = seg_paths
    exam.preferred_segmentation_path = seg_paths[0] if seg_paths else None
    return exam


def discover_exams(root_dir: str, settings: BatchDiscoverySettings, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> BatchDiscoveryResult:
    root = Path(root_dir)
    result = BatchDiscoveryResult(settings=settings, exams=[], skipped_dirs=[])
    if not root.exists():
        result.skipped_dirs.append({'dir': str(root), 'reason': 'root directory does not exist'})
        return result
    if not root.is_dir():
        result.skipped_dirs.append({'dir': str(root), 'reason': 'root path is not a directory'})
        return result

    all_dirs = []
    for current_root, dirs, _files in os.walk(root, topdown=True):
        all_dirs.append(current_root)
    total_dirs = max(len(all_dirs), 1)

    discovered_exam_dirs: set[str] = set()
    processed = 0
    for current_root, dirs, _files in os.walk(root, topdown=True):
        path = Path(current_root)
        processed += 1
        if progress_callback is not None:
            progress_callback(processed, total_dirs, f"Scanning: {path.name or str(path)}")
        is_exam, reason, exam = classify_directory_as_exam(path, settings)
        if is_exam and exam is not None:
            if exam.exam_dir not in discovered_exam_dirs:
                result.exams.append(exam)
                discovered_exam_dirs.add(exam.exam_dir)
            dirs[:] = []  # Do not descend into subfolders of exam folders.
        else:
            result.skipped_dirs.append({'dir': str(path), 'reason': reason})

    result.exams.sort(key=lambda e: (e.patient_id.lower(), e.exam_id.lower(), e.exam_dir.lower()))
    detected = get_detected_suffixes(result)
    if detected and not result.launch_selection.selected_suffix:
        result.launch_selection.selected_suffix = detected[0]
    if progress_callback is not None:
        progress_callback(total_dirs, total_dirs, 'Scan complete')
    return result


def get_detected_suffixes(result: BatchDiscoveryResult) -> list[str]:
    """Configured volume suffixes found in at least one exam, in settings order."""
    found = {sfx for exam in result.exams for sfx in exam.available_suffixes}
    return [sfx for sfx in result.settings.volume_suffixes if sfx in found]


def count_exams_with_suffix(result: BatchDiscoveryResult, suffix: str) -> int:
    return sum(1 for exam in result.exams if suffix in exam.available_suffixes)


def get_available_series_types_for_suffix(result: BatchDiscoveryResult, suffix: str) -> list[str]:
    series_types = set()
    for exam in result.exams:
        series_types.update(exam.available_series_types_by_suffix.get(suffix, []))
    return sorted(series_types)


def exam_has_all_selected_series(exam: DiscoveredExam, suffix: str, selected_series: list[str]) -> bool:
    if not selected_series:
        return True
    available = set(exam.available_series_types_by_suffix.get(suffix, []))
    return all(s in available for s in selected_series)


