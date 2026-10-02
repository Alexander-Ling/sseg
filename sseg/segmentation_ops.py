"""Segmentation-operation mixins for the sseg viewer."""

from collections import deque
from functools import partial
from queue import Queue

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, label
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog

from . import perf


class SegmentationOpsMixin:
    @staticmethod
    def bbox_of_mask(mask):
            """Tight bounding box of the True voxels of a 3D boolean array as ((x0,x1),(y0,y1),(z0,z1)), or None if empty."""
            box = []
            for axis in range(3):
                others = tuple(a for a in range(3) if a != axis)
                idx = np.flatnonzero(mask.any(axis=others))
                if idx.size == 0:
                    return None
                box.append((int(idx[0]), int(idx[-1]) + 1))
            return tuple(box)

    @staticmethod
    def union_bbox(a, b):
            if a is None:
                return b
            if b is None:
                return a
            return tuple((min(x[0], y[0]), max(x[1], y[1])) for x, y in zip(a, b))

    @staticmethod
    def bbox_slices(bbox):
            return tuple(slice(lo, hi) for lo, hi in bbox)

    @perf.timed("handle_segmentation_update")
    def handle_segmentation_update(self, new_segmentation, action=None, level=None, bbox=None):
            """Apply `new_segmentation` to the current segmentation.

            Only the bounding box that can change is touched, copied for undo, and re-rendered. `bbox`
            ((x0,x1),(y0,y1),(z0,z1)) may be passed by callers that already know it (e.g. brush strokes).
            """
            if action is None:
                action = self.new_segmentation_overlap
            if level is None:
                level = self.segmentation_level

            if not self.current_segmentation_key:
                return
            current_seg = self.current_segmentation

            if bbox is None:
                bbox = self.bbox_of_mask(new_segmentation != 0)
                if action == "replace_level":
                    # replace_level also clears the level wherever it currently exists
                    bbox = self.union_bbox(bbox, self.bbox_of_mask(current_seg == level))
                if bbox is None:
                    return  # nothing can change
            region = self.bbox_slices(bbox)

            self.push_to_undo_stack(bbox)

            cur = current_seg[region]  # view: assignments below modify current_seg in place
            new = new_segmentation[region]
            if self.roi_mask is not None:
                new = self.roi_mask[region] & (new != 0)

            if action == "add_around":
                mask = (cur == 0)  # Mask where there is no segmentation
                cur[mask] = new[mask] * level
            elif action == "overwrite":
                mask = (new != 0) & np.isin(cur, list(self.overwrite_levels))
                cur[mask] = level
            elif action == "overwrite_all":
                mask = (new != 0)
                cur[mask] = level
            elif action == "erase":
                mask = (new != 0)  # Mask where the new segmentation is applied
                cur[mask] = 0  # Erase overlapping areas
            elif action == "erase_level":
                mask = (new != 0) & (cur == level)  # Mask where new segmentation overlaps with the selected level
                cur[mask] = 0  # Erase only the selected level
            elif action == "replace_level":
                # Set the current level to 0
                cur[cur == level] = 0
                # Apply new segmentation to current level
                mask = (new != 0)
                cur[mask] = level
            elif action == "overlap_only":
                overlap_level = self.overlap_level_combo.currentText()
                if overlap_level != "All":
                    mask = (cur == int(overlap_level)) & (new == 1)
                    cur[mask] = level
                else:
                    mask = (cur != 0) & (new == 1)
                    cur[mask] = level

            self.segmentation_volumes[self.current_segmentation_key] = current_seg
            self.current_segmentation = current_seg  # Update the current segmentation
            self.mark_segmentation_dirty(self.current_segmentation_key, True)
            self.refresh_unique_levels(changed_region=region)
            self.update_level_menus_if_changed()
            self.update_visualization(reset_volume=False, dirty_bbox=bbox)
            self.display_all_slices()

    def get_gradient_parameters(self):
            # Fetch and convert the intensity and gradient percentages from QLineEdit to float
            try:
                intensity_percentage = float(self.intensity_percentage_line_edit.text())
                gradient_percentage = float(self.gradient_percentage_line_edit.text())
                max_ticks = int(self.max_ticks_line_edit.text())
            except ValueError:
                # Handle the case where conversion fails, perhaps setting defaults or showing an error
                QMessageBox.warning(self, "Parameter Error", "Invalid parameters--default values used. Please enter valid numbers.")
                return 0.05, 0.01, 50000

            return intensity_percentage, gradient_percentage, max_ticks

    def get_patch_contour_parameters(self):
            # Fetch and convert the intensity and gradient percentages from QLineEdit to float
            try:
                intensity_threshold = float(self.intensity_threshold_line_edit.text())
                max_iterations = int(self.max_iterations_line_edit.text())
                neighborhood_size = int(self.neighborhood_size_line_edit.text())
            except ValueError:
                # Handle the case where conversion fails, perhaps setting defaults or showing an error
                QMessageBox.warning(self, "Parameter Error", "Invalid parameters--default values used. Please enter valid numbers.")
                return {
                            'intensity_threshold': 1.0,
                            'max_iterations': 1000000,
                            'neighborhood_size': 1
                        }

            return {
                        'intensity_threshold': intensity_threshold,
                        'max_iterations': max_iterations,
                        'neighborhood_size': neighborhood_size
                    }

    def execute_patch_based_region_growth(self, seed_mask):        
            # Define parameters for the region growing algorithm
            parameters = self.get_patch_contour_parameters()
            new_segmentation = self.patch_based_region_growth(seed_mask, parameters)
            # Update the segmentation volume
            self.handle_segmentation_update(new_segmentation)
            return new_segmentation

    def init_2d_brush_cache(self):
            orientation = self.get_orientation_by_view(self.current_view)
            self.brush_slice_cache = self.get_current_slice_data(orientation)
            seg_slice_data = self.get_current_segmentation_slice_data(orientation)

            self.unique_levels = np.union1d(self.unique_levels, self.segmentation_level)

            # Initialize the cache for slice data and segmentation slice data
            if seg_slice_data is None:
                # Initialize an empty array if the current segmentation slice data is None
                self.current_seg_slice_cache_2d = np.zeros_like(self.brush_slice_cache, dtype=int)
            else:
                self.current_seg_slice_cache_2d = seg_slice_data
            self.brush_cache_2d = np.zeros_like(self.brush_slice_cache, dtype=int)
            self._last_brush_point = None

    def _brush_disk(self, radius):
            cache = self.__dict__.setdefault('_brush_disk_cache', {})
            disk = cache.get(radius)
            if disk is None:
                offsets = np.arange(-radius, radius + 1)
                disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius ** 2
                cache[radius] = disk
            return disk

    def _stamp_brush(self, disk, radius, y, x):
            rows, cols = self.brush_cache_2d.shape
            y0, y1 = max(0, y - radius), min(rows, y + radius + 1)
            x0, x1 = max(0, x - radius), min(cols, x + radius + 1)
            if y0 >= y1 or x0 >= x1:
                return
            sub = disk[y0 - (y - radius):y1 - (y - radius), x0 - (x - radius):x1 - (x - radius)]
            self.brush_cache_2d[y0:y1, x0:x1][sub] = 1

    @perf.timed("apply_2d_brush")
    def apply_2d_brush(self, coord):
            if self.current_segmentation is None or not self.is_drawing_2d_brush:
                return

            brush_size = self.brush_size_slider.value()
            half_brush_size = brush_size // 2
            orientation = self.get_orientation_by_view(self.current_view)

            # Extract x, y from coord depending on orientation
            if orientation == 'axial':
                y, x, slice_index = coord
            elif orientation == 'coronal':
                y, slice_index, x = coord
            elif orientation == 'sagittal':
                slice_index, y, x = coord

            # Stamp a precomputed disk along the path from the previous mouse position, so fast
            # strokes leave no gaps. (The disk is cached per radius; a point is inside if its distance <= radius.)
            disk = self._brush_disk(half_brush_size)
            last = getattr(self, '_last_brush_point', None)
            if last is not None and last[0] == orientation:
                steps = max(abs(y - last[1]), abs(x - last[2]), 1)
                ys = np.rint(np.linspace(last[1], y, steps + 1)).astype(int)
                xs = np.rint(np.linspace(last[2], x, steps + 1)).astype(int)
            else:
                ys, xs = np.array([y]), np.array([x])
            for py, px in zip(ys, xs):
                self._stamp_brush(disk, half_brush_size, int(py), int(px))
            self._last_brush_point = (orientation, y, x)

            # Update the overlay in the viewer
            self.update_2d_brush_highlights(orientation)

    def finalize_2d_brush_strokes(self, returnROI=False):
            orientation_to_index = {'axial': 2, 'coronal': 1, 'sagittal': 0}
            if self.current_segmentation is None or not self.is_drawing_2d_brush:
                return

            orientation = self.get_orientation_by_view(self.current_view)
            if orientation == 'axial':
                y, x, slice_index = self.coord
                max_slice = self.data.shape[2]
            elif orientation == 'coronal':
                y, slice_index, x = self.coord
                max_slice = self.data.shape[1]
            elif orientation == 'sagittal':
                slice_index, y, x = self.coord
                max_slice = self.data.shape[0]

            new_segmentation = np.zeros(self.current_segmentation.shape, dtype=bool)
            thickness = self.brush_thickness_slider.value()

            if thickness % 2 == 0:
                even_thickness_modifier = 1
                thickness = thickness - 1
            else:
                even_thickness_modifier = 0

            if self.brush_position == "Centered":
                start_slice = max((slice_index - thickness // 2) - even_thickness_modifier, 0)
                end_slice = min(slice_index + thickness // 2 + 1, max_slice)
            elif self.brush_position == "Above":
                start_slice = slice_index
                end_slice = min(slice_index + thickness + even_thickness_modifier, max_slice)
            elif self.brush_position == "Below":
                start_slice = max(slice_index - thickness - even_thickness_modifier + 1, 0)
                end_slice = slice_index + 1

            # Assign the cached brush strokes to the relevant slices
            if orientation == 'axial':
                new_segmentation[:, :, start_slice:end_slice] = np.tile(self.brush_cache_2d[:, :, np.newaxis], (1, 1, end_slice - start_slice))
            elif orientation == 'coronal':
                new_segmentation[:, start_slice:end_slice, :] = np.tile(self.brush_cache_2d[:, np.newaxis, :], (1, end_slice - start_slice, 1))
            elif orientation == 'sagittal':
                new_segmentation[start_slice:end_slice, :, :] = np.tile(self.brush_cache_2d[np.newaxis, :, :], (end_slice - start_slice, 1, 1))

            # Bounding box of the painted slab (in-plane extent from the 2D cache), so the update only touches it
            bbox = None
            rows = np.flatnonzero(self.brush_cache_2d.any(axis=1))
            cols = np.flatnonzero(self.brush_cache_2d.any(axis=0))
            if rows.size and cols.size and end_slice > start_slice:
                r, c, sl = (int(rows[0]), int(rows[-1]) + 1), (int(cols[0]), int(cols[-1]) + 1), (start_slice, end_slice)
                if orientation == 'axial':
                    bbox = (r, c, sl)
                elif orientation == 'coronal':
                    bbox = (r, sl, c)
                else:
                    bbox = (sl, r, c)

            # Cleaning up caches
            self.brush_cache_2d = None
            self.current_seg_slice_cache_2d = None
            self.brush_slice_cache = None

            # Either returning painted volume or sending to update segmentation directly     
            if returnROI:
                return new_segmentation
            else:
                self.handle_segmentation_update(new_segmentation, bbox=bbox)
                return None

    def modify_segmentation_borders(self, change_value, action = "grow"):
            if self.current_segmentation is None:
                return
            # One call to handle_segmentation_update per operation, so each Grow/Shrink/Hollow is a single undo step.
            segmentation_mask = self.current_segmentation == self.segmentation_level
            if action == "grow":
                modified_mask = binary_dilation(segmentation_mask, iterations=change_value)
                self.handle_segmentation_update(modified_mask)
            elif action == "shrink":
                modified_mask = binary_erosion(segmentation_mask, iterations=change_value)
                self.handle_segmentation_update(modified_mask, action = "replace_level")
            elif action == "hollow":
                modified_mask = binary_erosion(segmentation_mask, iterations=change_value)
                self.handle_segmentation_update(modified_mask, action = "erase_level")

    def find_contiguous_volume(self, start_coord, level):
            """Boolean mask of the 6-connected region of voxels equal to `level` that contains `start_coord`."""
            labels, _ = label(self.current_segmentation == level)  # default structure: 6-connectivity in 3D
            seed_label = labels[tuple(start_coord)]
            if seed_label == 0:
                return np.zeros(self.current_segmentation.shape, dtype=bool)
            return labels == seed_label

    def apply_remove_contiguous_volume(self):
            if self.coord:
                z, y, x = self.coord
                level = self.current_segmentation[z, y, x]
                if not level == 0:
                    mask = self.find_contiguous_volume((z, y, x), level)
                    self.handle_segmentation_update(mask, action="erase")

    def apply_keep_only_contiguous_volume(self):
            if self.coord:
                z, y, x = self.coord
                level = self.current_segmentation[z, y, x]
                if not level == 0:
                    mask = self.find_contiguous_volume((z, y, x), level)
                    self.handle_segmentation_update(mask, action="replace_level", level = level)

    def apply_change_volume_level(self):
            if self.coord:
                z, y, x = self.coord
                current_level = self.current_segmentation[z, y, x]
                if current_level != 0:
                    new_level = self.segmentation_level
                    mask = self.find_contiguous_volume((z, y, x), current_level)
                    self.handle_segmentation_update(mask, action="overwrite_all", level=new_level)

    def apply_fill_contiguous_volume_holes(self):
        if self.coord:
            z, y, x = self.coord
            level = self.current_segmentation[z, y, x]
            if level != 0:
                mask = self.find_contiguous_volume((z, y, x), level)
                filled_mask = binary_fill_holes(mask)
                new_voxels = filled_mask & (~mask) & (self.current_segmentation == 0)
                self.handle_segmentation_update(new_voxels, action="overwrite_all", level=level)

    def gradient_tracing(self, seed_point, base_value, intensity_percentage=0.05, gradient_percentage=0.01, max_ticks=50000):
            """
            Perform gradient tracing to identify a 3D region around the seed_point where
            the intensity is similar to the base_value and the gradient does not exceed
            the gradient_threshold based on percentage thresholds.

            Parameters:
                seed_point (tuple): The (z, y, x) coordinates of the seed voxel.
                base_value (float): The intensity value of the seed voxel.
                intensity_percentage (float): Percentage of the total intensity range to define the intensity threshold.
                gradient_percentage (float): Percentage of the total intensity range to define the gradient threshold.
                max_ticks (int): Maximum number of iterations to run the tracing for performance control.

            Returns:
                np.array: A binary mask of the same shape as self.data where the region is 1 and elsewhere is 0.
            """
            # Validate seed_point and base_value
            if seed_point is None or base_value is None:
                QMessageBox.warning(self, "Gradient Tracing", "Seed point or base value is invalid.")
                return np.zeros_like(self.data, dtype=bool)

            if not (0 <= seed_point[0] < self.data.shape[0] and 0 <= seed_point[1] < self.data.shape[1] and 0 <= seed_point[2] < self.data.shape[2]):
                QMessageBox.warning(self, "Gradient Tracing", "Seed point is out of bounds.")
                return np.zeros_like(self.data, dtype=bool)

            # Initialize the ROI mask
            if self.roi_mask is not None:
                if self.roi_mask.shape != self.data.shape:
                    QMessageBox.warning(self, "Gradient Tracing", "ROI mask dimensions do not match the data dimensions.")
                    return np.zeros_like(self.data, dtype=bool)
                roi_mask = self.roi_mask.astype(bool)
            else:
                roi_mask = np.ones_like(self.data, dtype=bool)

            # Check if seed_point is within ROI
            if not roi_mask[seed_point]:
                QMessageBox.warning(self, "Gradient Tracing", "Seed point is outside the active ROI mask.")
                return np.zeros_like(self.data, dtype=bool)

            # Calculate thresholds
            min_val = np.min(self.data)
            max_val = np.max(self.data)
            total_range = max_val - min_val
            intensity_threshold = intensity_percentage * total_range
            gradient_threshold = gradient_percentage * total_range

            # Initialize the mask and process list
            mask = np.zeros_like(self.data, dtype=bool)
            mask[seed_point] = True
            process_list = deque([tuple(int(c) for c in seed_point)])

            connectivity = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1)
            ]

            progress_dialog = QProgressDialog("Processing gradient tracing...", "Abort", 0, max_ticks, self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(500)

            # Main processing loop
            tick = 0
            while process_list and tick < max_ticks:
                if progress_dialog.wasCanceled():
                    QMessageBox.information(self, "Gradient Tracing", "Operation aborted by user.")
                    return np.zeros_like(self.data, dtype=bool)

                current_point = process_list.popleft()
                current_intensity = self.data[current_point]

                for conn in connectivity:
                    neighbor = (current_point[0] + conn[0], current_point[1] + conn[1], current_point[2] + conn[2])

                    # Ensure neighbor is within bounds
                    if all(0 <= n < s for n, s in zip(neighbor, self.data.shape)):
                        neighbor_intensity = self.data[neighbor]
                        intensity_diff = abs(neighbor_intensity - base_value)
                        gradient = abs(neighbor_intensity - current_intensity)

                        # Check if the neighbor is valid
                        if not roi_mask[neighbor] or mask[neighbor]:
                            continue  # Skip already processed or out-of-ROI points

                        # Apply intensity and gradient thresholds
                        if intensity_diff <= intensity_threshold and gradient <= gradient_threshold:
                            mask[neighbor] = True
                            process_list.append(neighbor)

                tick += 1
                if tick % 1000 == 0:  # updating the dialog every voxel dominated the run time
                    progress_dialog.setValue(tick)
                    QApplication.processEvents()

            progress_dialog.close()

            if tick >= max_ticks:
                QMessageBox.warning(self, "Gradient Tracing", "Maximum iterations reached. Tracing stopped early.")

            return mask

    def gradient_tracing_global(self, seed_point, base_value, intensity_percentage=0.05, gradient_percentage=0.01, max_ticks=50000):
            """
            Perform global gradient tracing and segmentation based on the value range
            calculated from the traced region.
            """
            # Perform local gradient tracing to get the mask
            mask = self.gradient_tracing(seed_point, base_value, intensity_percentage, gradient_percentage, max_ticks)

            # Extract values in the traced region
            data_values = self.data[mask]

            # Check if data_values is empty
            if data_values.size == 0:
                QMessageBox.warning(self, "Gradient Tracing Error", "No valid voxels were found during tracing.")
                return np.zeros_like(self.data, dtype=np.bool_)  # Return an empty segmentation mask

            # Calculate the value range in the traced region
            min_val, max_val = data_values.min(), data_values.max()

            # Segment globally based on the value range
            global_segmentation = (self.data >= min_val) & (self.data <= max_val)

            return global_segmentation

    def patch_based_region_growth(self, seed_mask, parameters):
        
            if seed_mask is None or seed_mask.ndim != 3:
                QMessageBox.warning(
                    self,
                    "Region Growth Error",
                    "The seed mask is empty, invalid, or could not be created. Returning an empty mask."
                )
                # Return an empty mask with the same shape as the MRI data
                empty_mask = np.zeros_like(self.data, dtype=np.uint8)
                return empty_mask
        
            if seed_mask.dtype != bool:
                seed_mask = seed_mask.astype(bool)

            # Initialize the ROI mask
            if self.roi_mask is not None:
                roi_mask = self.roi_mask.astype(bool)
            else:
                roi_mask = np.ones_like(self.data, dtype=bool)
        
            # Extract intensity values of the ROI
            roi_values = self.data[seed_mask & roi_mask] #Note that this restricts seed_mask to roi mask region if a segmentation ROI has been defined.
        
            if roi_values.size == 0:
                if self.roi_mask is not None:
                    QMessageBox.warning(self, "Region Growing", "Seed mask is empty. Please select a valid region within the active ROI mask.")
                else:
                    QMessageBox.warning(self, "Region Growing", "Seed mask is empty. Please select a valid region.")
                return seed_mask  # Return the original mask
        
            # Compute summary metrics
            roi_mean = roi_values.mean()
            roi_std = roi_values.std()
        
            # Set thresholds from parameters
            intensity_threshold = parameters.get('intensity_threshold', 1.0)  # Default to 1.0 times the ROI std
            max_iterations = parameters.get('max_iterations', 1000000)
            neighborhood_size = parameters.get('neighborhood_size', 1)  # Default neighborhood size is 1
                
            # Fast path: the grown region is the set of acceptable voxels connected (6-connectivity) to the seeds,
            # which connected-component labelling computes directly. It is exactly equivalent to the loop below
            # whenever the iteration cap is not reached; otherwise fall through to the original loop.
            acceptable = roi_mask & (np.abs(self.data - roi_mean) <= intensity_threshold * roi_std)
            touching = binary_dilation(seed_mask) | seed_mask
            labels, _ = label(acceptable)
            wanted = np.unique(labels[touching & acceptable])
            wanted = wanted[wanted != 0]
            fast_mask = (seed_mask & roi_mask) | (np.isin(labels, wanted) if wanted.size else False)
            added = int(fast_mask.sum()) - int((seed_mask & roi_mask).sum())
            if int(seed_mask.sum()) + max(added, 0) <= max_iterations:
                return fast_mask

            # Initialize the mask and the processed array as boolean arrays
            mask = seed_mask.copy()
            mask &= roi_mask  # Ensure seed_mask is within ROI
            processed = mask.copy()
        
            # Initialize the queue with seed points
            seed_points = np.argwhere(seed_mask)
        
            # Convert seed_points to deque of tuples for efficient FIFO operations
            queue = deque(map(tuple, seed_points))
        
            # Define connectivity (6-connected neighborhood)
            directions = [
                (dz, dy, dx)
                for dz in range(-neighborhood_size, neighborhood_size + 1)
                for dy in range(-neighborhood_size, neighborhood_size + 1)
                for dx in range(-neighborhood_size, neighborhood_size + 1)
                if abs(dz) + abs(dy) + abs(dx) == 1  # Ensures only immediate neighbors are considered
            ]
        
            iterations = 0
        
            # Set up the progress dialog
            progress_dialog = QProgressDialog("Performing patch-based region growth...", "Abort", 0, max_iterations, self)
            progress_dialog.setWindowTitle("Region Growing")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(500)  # Show dialog after 500 ms
            progress_dialog.setValue(0)
        
            while queue and iterations < max_iterations:
                if progress_dialog.wasCanceled():
                    break
        
                current_point = queue.popleft()
                z, y, x = current_point
                current_intensity = self.data[z, y, x]
        
                for dz, dy, dx in directions:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    # Check bounds
                    if 0 <= nz < self.data.shape[0] and 0 <= ny < self.data.shape[1] and 0 <= nx < self.data.shape[2]:
                        if not processed[nz, ny, nx] and roi_mask[nz, ny, nx]:
                            neighbor_intensity = self.data[nz, ny, nx]
                            intensity_diff = abs(neighbor_intensity - roi_mean)
        
                            if intensity_diff <= intensity_threshold * roi_std:
                                mask[nz, ny, nx] = True
                                queue.append((nz, ny, nx))
                            processed[nz, ny, nx] = True
        
                iterations += 1
                if iterations % 1000 == 0 or iterations == max_iterations:
                    progress_dialog.setValue(iterations)
                    QApplication.processEvents()
        
            progress_dialog.setValue(iterations)
            progress_dialog.close()
        
            if iterations >= max_iterations:
                QMessageBox.information(self, "Region Growing", "Maximum iterations reached during region growing.")
        
            return mask
