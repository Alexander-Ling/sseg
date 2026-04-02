from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import nibabel as nib
import nrrd
import numpy as np
from matplotlib import colors as mcolors


@dataclass
class LoadedVolume:
    data: np.ndarray
    affine: np.ndarray
    voxel_dims: tuple


def determine_nrrd_space(affine):
    pos_labels = ["right", "anterior", "superior"]
    neg_labels = ["left", "posterior", "inferior"]
    orientation = []
    for i in range(3):
        column = affine[:3, i]
        major_axis_index = np.argmax(np.abs(column))
        axis_sign = np.sign(column[major_axis_index])
        orientation.append(pos_labels[major_axis_index] if axis_sign > 0 else neg_labels[major_axis_index])
    return "-".join(orientation)


def reorient_affine_space_to_ras(affine, current_space):
    space_mapping = {
        "left-anterior-superior": "LAS",
        "left-anterior-inferior": "LAI",
        "left-posterior-superior": "LPS",
        "left-posterior-inferior": "LPI",
        "right-anterior-superior": "RAS",
        "right-anterior-inferior": "RAI",
        "right-posterior-superior": "RPS",
        "right-posterior-inferior": "RPI",
    }
    if current_space.lower() not in space_mapping:
        raise ValueError(f"Unknown current_space provided: {current_space}")
    current_orientation = space_mapping[current_space.lower()]
    transform_matrix = np.eye(4)
    for i, axis in enumerate(current_orientation):
        if axis == "L":
            transform_matrix[i, i] = -1
        elif axis == "P":
            transform_matrix[1, i] = -1
        elif axis == "I":
            transform_matrix[2, i] = -1
    return np.dot(affine, transform_matrix)


def load_segmentation_data(file_path: str):
    level_names: Dict[int, str] = {}
    if file_path.endswith('.nrrd'):
        data, header = nrrd.read(file_path)
        affine = np.eye(4)
        if 'space directions' in header and 'space origin' in header:
            space_directions = np.array(header['space directions'])
            if space_directions.shape == (4, 3):
                affine[:3, :3] = space_directions[1:, :]
            else:
                affine[:3, :3] = space_directions
            affine[:3, 3] = np.array(header['space origin'])
        affine = reorient_affine_space_to_ras(affine, header['space'])
        img = nib.as_closest_canonical(nib.Nifti1Image(data.astype(np.int32), affine))
        segment_count = 0
        while True:
            segment_name_key = f"Segment{segment_count}_Name"
            segment_label_key = f"Segment{segment_count}_LabelValue"
            if segment_name_key in header and segment_label_key in header:
                level_name = header[segment_name_key]
                label_value = int(header[segment_label_key])
                level_names[label_value] = level_name
                segment_count += 1
            else:
                break
    else:
        img = nib.as_closest_canonical(nib.load(file_path))
    return img.get_fdata().astype(np.int32), level_names


def load_and_reorient_volume(file_path: str, *, master_affine=None, master_orientation=None,
                             master_voxel_dims=None, master_shape=None, master_canonical_affine=None,
                             master_canonical_orientation=None, master_canonical_voxel_dims=None,
                             master_canonical_shape=None):
    img = nib.load(file_path)

    if master_affine is None:
        master_affine = img.affine
    if master_orientation is None:
        master_orientation = determine_nrrd_space(master_affine)
    if master_voxel_dims is None:
        master_voxel_dims = img.header.get_zooms()
    if master_shape is None:
        master_shape = img.get_fdata().shape

    ras_img = nib.as_closest_canonical(img)
    data = ras_img.get_fdata()
    affine = ras_img.affine
    orientation = determine_nrrd_space(affine)
    voxel_dims = ras_img.header.get_zooms()

    if master_canonical_affine is None:
        master_canonical_affine = affine
    if master_canonical_orientation is None:
        master_canonical_orientation = determine_nrrd_space(affine)
    if master_canonical_voxel_dims is None:
        master_canonical_voxel_dims = voxel_dims
    if master_canonical_shape is None:
        master_canonical_shape = data.shape

    if not np.allclose(master_canonical_affine, affine, atol=1e-4):
        raise ValueError(
            f"Attempted to load volume {file_path} with RAS affine that does not match previously loaded RAS affine."
        )
    if not np.array_equal(master_canonical_orientation, orientation):
        raise ValueError(
            f"During attempt to load volume {file_path}, canonical orientation = {master_canonical_orientation}; loaded orientation = {orientation}."
        )
    if not np.allclose(master_canonical_voxel_dims, voxel_dims, atol=1e-3):
        raise ValueError(
            f"Attempted to load volume {file_path} with RAS voxel dimensions {voxel_dims} that do not match {master_canonical_voxel_dims}."
        )
    if not np.array_equal(master_canonical_shape, data.shape):
        raise ValueError(
            f"Attempted to load volume {file_path} with RAS shape {data.shape} that does not match {master_canonical_shape}."
        )

    return LoadedVolume(data=data, affine=affine, voxel_dims=voxel_dims)


def reorient_to_original(data, *, master_canonical_affine, master_affine):
    ras_image = nib.Nifti1Image(data, master_canonical_affine)
    ras_ornt = nib.orientations.io_orientation(master_canonical_affine)
    original_ornt = nib.orientations.io_orientation(master_affine)
    ornt_transform = nib.orientations.ornt_transform(ras_ornt, original_ornt)
    original_oriented_data = nib.orientations.apply_orientation(ras_image.get_fdata(), ornt_transform)
    return nib.Nifti1Image(original_oriented_data, master_affine)


def get_rgb(color_name):
    rgb = mcolors.to_rgb(color_name)
    return ' '.join(map(str, rgb))


def default_save_directory(volume_paths, segmentation_paths):
    if volume_paths and os.path.exists(os.path.dirname(volume_paths[0])):
        return os.path.dirname(volume_paths[0])
    if segmentation_paths and os.path.exists(os.path.dirname(segmentation_paths[0])):
        return os.path.dirname(segmentation_paths[0])
    return os.path.expanduser('~')


def save_segmentation_nifti(data, file_path, *, master_canonical_affine, master_affine):
    img = reorient_to_original(data.astype(np.int32), master_canonical_affine=master_canonical_affine, master_affine=master_affine)
    nib.save(img, file_path)


def save_segmentation_nrrd(data, file_path, *, master_canonical_affine, master_affine, unique_levels, level_names: Optional[dict] = None):
    img = reorient_to_original(data.astype(np.int32), master_canonical_affine=master_canonical_affine, master_affine=master_affine)
    affine = img.affine
    out = img.get_fdata()
    header = {}
    header['type'] = 'int'
    header['dimension'] = 3
    header['space'] = 'right-anterior-superior'
    header['sizes'] = f'array({list(out.shape)})'
    header['kinds'] = ['domain', 'domain', 'domain']
    header['space directions'] = affine[:3, :3].tolist()
    header['space origin'] = affine[:3, 3].tolist()

    names = level_names or {}
    for idx, level in enumerate(level for level in unique_levels if level != 0):
        name = names.get(level, f'Segment_{level}')
        header[f'Segment{idx}_ID'] = f'Segment_{level}'
        header[f'Segment{idx}_LabelValue'] = int(level)
        header[f'Segment{idx}_Layer'] = 0
        header[f'Segment{idx}_Name'] = name
        header[f'Segment{idx}_NameAutoGenerated'] = 0 if level in names else 1

    nrrd.write(file_path, out, header)
