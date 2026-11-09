from __future__ import annotations

import os
import re
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


def segmentation_type_for_suffix(suffix: str) -> str:
    """'_tumor-segmentation.nii.gz' -> '_tumor-segmentation'; suffixes differing only by extension share a type."""
    return strip_known_image_extension(suffix)


def get_segmentation_types(settings: BatchDiscoverySettings) -> list[str]:
    """Distinct segmentation types from the configured suffixes, in settings order."""
    types: list[str] = []
    for sfx in settings.segmentation_suffixes:
        seg_type = segmentation_type_for_suffix(sfx) if sfx else ''
        if seg_type and seg_type not in types:
            types.append(seg_type)
    return types


def match_segmentation_suffix(filename: str, segmentation_suffixes: list[str]) -> Optional[str]:
    """Return the configured segmentation suffix that ends the filename (longest wins), else None."""
    matches = [sfx for sfx in segmentation_suffixes if sfx and segmentation_type_for_suffix(sfx) and filename.endswith(sfx)]
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


# Integers, with a leading '-' treated as a minus sign unless it follows a digit
# (so 'd-12' is -12, but the dashes in '2020-01-05' are separators).
_NATURAL_NUMBER_RE = re.compile(r'((?<![0-9])-[0-9]+|[0-9]+)')


def natural_sort_key(text: str) -> tuple:
    """Sort key that orders embedded numbers numerically: d-203, d-1, d0, d29, d120."""
    # re.split with a capture group alternates text, number, text, ..., so the
    # str/int positions line up between keys and tuple comparison never mixes types.
    parts = _NATURAL_NUMBER_RE.split(text)
    return tuple(int(part) if i % 2 else part.casefold() for i, part in enumerate(parts))


def exam_sort_key(exam: DiscoveredExam) -> tuple:
    return (natural_sort_key(exam.patient_id), natural_sort_key(exam.exam_id), natural_sort_key(exam.exam_dir))


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
    for base_dir in (exam_dir, patient_dir):
        if not base_dir.is_dir():
            continue
        for sfx in settings.brainmask_suffixes:
            matches = sorted(p for p in base_dir.iterdir() if p.is_file() and p.name.endswith(sfx))
            if matches:
                exam.brainmask_path = str(matches[0])
                break
        if exam.brainmask_path is not None:
            break

    # Group segmentation files by type, in settings order. Each file belongs only to the type of
    # its longest matching suffix, so overlapping suffixes never claim the same file twice.
    by_type: dict[str, list[str]] = {seg_type: [] for seg_type in get_segmentation_types(settings)}
    for base_dir in (exam_dir, patient_dir):
        for p in sorted(base_dir.iterdir() if base_dir.is_dir() else []):
            if not p.is_file():
                continue
            sfx = match_segmentation_suffix(p.name, settings.segmentation_suffixes)
            if sfx is not None:
                by_type[segmentation_type_for_suffix(sfx)].append(str(p))
    exam.segmentation_paths_by_type = {seg_type: paths for seg_type, paths in by_type.items() if paths}
    seg_paths = [p for paths in exam.segmentation_paths_by_type.values() for p in paths]
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

    result.exams.sort(key=exam_sort_key)
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


def get_detected_segmentation_types(result: BatchDiscoveryResult) -> list[str]:
    """Segmentation types found in at least one exam, in settings order."""
    found = {seg_type for exam in result.exams for seg_type in exam.segmentation_paths_by_type}
    return [seg_type for seg_type in get_segmentation_types(result.settings) if seg_type in found]


def count_exams_with_segmentation_type(result: BatchDiscoveryResult, seg_type: str) -> int:
    return sum(1 for exam in result.exams if seg_type in exam.segmentation_paths_by_type)


def get_exam_segmentation_paths(exam: DiscoveredExam, selected_types: Optional[list[str]]) -> list[str]:
    """Segmentation files to load for an exam. selected_types=None means every type.

    Files with the same name (e.g. one in the exam folder and one in the patient folder)
    are loaded once, preferring the exam folder copy.
    """
    paths = []
    names = set()
    for seg_type, type_paths in exam.segmentation_paths_by_type.items():
        if selected_types is not None and seg_type not in selected_types:
            continue
        for path in type_paths:
            name = Path(path).name
            if name not in names:
                paths.append(path)
                names.add(name)
    return paths


def exam_matches_segmentation_filter(exam: DiscoveredExam, seg_types: list[str], require_all: bool, require_any: bool) -> bool:
    """Whether an exam passes the 'all selected' / 'at least one selected' segmentation filters.

    seg_types should already be resolved (None -> every detected type) by the caller.
    An empty selection requires nothing.
    """
    if not seg_types:
        return True
    present = [seg_type in exam.segmentation_paths_by_type for seg_type in seg_types]
    if require_all:
        return all(present)
    if require_any:
        return any(present)
    return True


def exam_has_all_selected_series(exam: DiscoveredExam, suffix: str, selected_series: list[str]) -> bool:
    if not selected_series:
        return True
    available = set(exam.available_series_types_by_suffix.get(suffix, []))
    return all(s in available for s in selected_series)


