from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


DEFAULT_IMAGE_EXTENSIONS = [".nii", ".nii.gz", ".nrrd"]
DEFAULT_BRAINMASK_SUFFIXES = [
    "_brainmask.nii.gz", "_brainmask.nii", "_brainmask.nrrd",
    "_Brainmask.nii.gz", "_Brainmask.nii", "_Brainmask.nrrd",
]
DEFAULT_SEGMENTATION_SUFFIXES = [
    "_tumor-segmentation.nii.gz", "_tumor-segmentation.nii", "_tumor-segmentation.nrrd",
    "_GBM-seg.nii.gz", "_GBM-seg.nii", "_GBM-seg.nrrd",
]
DEFAULT_VOLUME_SUFFIXES = ["_brain-norm.nii.gz"]


@dataclass
class SeriesFileInfo:
    path: str
    filename: str
    series_type: str
    shared_suffix: str
    compatible_group_key: str
    shape: list[int] = field(default_factory=list)
    voxel_dims: list[float] = field(default_factory=list)


@dataclass
class DiscoveredExam:
    patient_id: str
    exam_id: str
    patient_dir: str
    exam_dir: str
    series_volume_paths: list[str] = field(default_factory=list)
    excluded_nonseries_paths: list[str] = field(default_factory=list)
    subdirs_present: list[str] = field(default_factory=list)
    series_files: list[SeriesFileInfo] = field(default_factory=list)
    available_suffixes: list[str] = field(default_factory=list)
    available_series_types_by_suffix: dict[str, list[str]] = field(default_factory=dict)
    compatible_groups: dict[str, list[str]] = field(default_factory=dict)
    preferred_group_by_suffix: dict[str, str] = field(default_factory=dict)
    brainmask_path: Optional[str] = None
    segmentation_paths: list[str] = field(default_factory=list)
    preferred_segmentation_path: Optional[str] = None


@dataclass
class BatchDiscoverySettings:
    root_dir: str = ""
    image_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))
    brainmask_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_BRAINMASK_SUFFIXES))
    segmentation_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_SEGMENTATION_SUFFIXES))
    verify_scan_compatibility: bool = False
    # Full filename endings (including extension) that identify displayable MRI volumes.
    # The text before the suffix, after its last '_', is the series type.
    volume_suffixes: list[str] = field(default_factory=lambda: list(DEFAULT_VOLUME_SUFFIXES))


@dataclass
class BatchLaunchSelection:
    selected_suffix: Optional[str] = None
    selected_series_types: list[str] = field(default_factory=list)
    require_all_selected_series: bool = False
    replacement_segmentation_suffix: str = ""


@dataclass
class BatchDiscoveryResult:
    settings: BatchDiscoverySettings
    exams: list[DiscoveredExam] = field(default_factory=list)
    skipped_dirs: list[dict[str, Any]] = field(default_factory=list)
    launch_selection: BatchLaunchSelection = field(default_factory=BatchLaunchSelection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchDiscoveryResult":
        known = BatchDiscoverySettings.__dataclass_fields__
        settings = BatchDiscoverySettings(**{k: v for k, v in data.get("settings", {}).items() if k in known})
        exams = []
        for item in data.get("exams", []):
            item = dict(item)
            sf = [SeriesFileInfo(**sf_item) for sf_item in item.get("series_files", [])]
            item["series_files"] = sf
            exams.append(DiscoveredExam(**item))
        skipped = list(data.get("skipped_dirs", []))
        launch_sel = BatchLaunchSelection(**data.get("launch_selection", {}))
        return cls(settings=settings, exams=exams, skipped_dirs=skipped, launch_selection=launch_sel)
