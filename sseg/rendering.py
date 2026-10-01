"""Rendering helpers and view update mixins for the sseg viewer."""

import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import ListedColormap, Normalize
from pyvista import Plane
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QAction, QGraphicsEllipseItem, QGraphicsScene
from vtk.util import numpy_support
from scipy.ndimage import binary_fill_holes

from .dialogs import ParameterDialog


class RenderingMixin:

    def build_foreground_mask(self, data):
            mask = np.abs(data) > 1e-3
            if np.any(mask):
                try:
                    mask = binary_fill_holes(mask)
                except Exception:
                    pass
            return mask

    def compute_auto_opacity_transfer_function(self, data):
            mask = self.build_foreground_mask(data)
            foreground = data[mask] if np.any(mask) else data[np.isfinite(data)]
            if foreground.size == 0:
                return np.zeros(256, dtype=float)

            levels = np.linspace(float(np.min(data)), float(np.max(data)), 256)
            low = float(np.percentile(foreground, 55))
            high = float(np.percentile(foreground, 99.5))
            if not np.isfinite(low):
                low = float(np.min(foreground))
            if not np.isfinite(high):
                high = float(np.max(foreground))

            if high <= low:
                high = low + max(abs(low) * 0.05, 1e-3)

            t = (levels - low) / (high - low)
            t = np.clip(t, 0.0, 1.0)
            # Keep the default auto rendering fairly conservative so normalized scans
            # do not appear as a fully opaque black block.
            opacities = 0.35 * (t ** 2.2)
            return opacities

    def display_placeholder_views(self, message="No exam loaded"):
            for orientation, view in self.views.items():
                scene = QGraphicsScene()
                text_item = scene.addText(message)
                text_item.setDefaultTextColor(QColor("white"))
                view.setScene(scene)
                view.setBackgroundBrush(QColor("black"))
                view.setSceneRect(scene.itemsBoundingRect())
                view.centerOn(text_item)

    def clear_3d_planes(self):
            for key in list(self.plane_meshes.keys()):
                try:
                    self.plotter_3d.remove_actor(self.plane_meshes[key][1])
                except Exception:
                    pass
                del self.plane_meshes[key]

    def create_3d_volume(self, data, voxel_dims):
            if data is None:
                self.plotter_3d.clear()
                self.clear_3d_planes()
                return
            # Create a vtkImageData object
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(data.shape)
            #vtk_data.SetSpacing(voxel_dims)

            # Normalize data if it needs to be normalized.
            if np.isclose(self.global_min, 0, atol=1e-3):
                data = self.normalize_data(data)

            # Convert the numpy array to a format compatible with VTK
            flat_data_array = data.flatten(order='F')
            vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_data.GetPointData().SetScalars(vtk_array)

            # Use the vtkImageData object to create a PyVista ImageData object
            grid = pv.ImageData(vtk_data)
        
            # Clear the existing plotter data and add the new volume
            self.plotter_3d.clear()

            # Auto 3D opacity is currently deprecated; keep the fallback path
            # available for backwards compatibility, but default to manual.
            if getattr(self, "auto_3d_opacity", False):
                opacities = self.compute_auto_opacity_transfer_function(data)
            else:
                opacities = self.biquadratic(data, self.steepness, self.exponent, self.opacity_multiplier)

            if self.new_camera:
                self.plotter_3d.add_volume(grid, mapper="gpu", cmap="gray", opacity=opacities, show_scalar_bar=False)
                self.new_camera = False
            else:
                self.plotter_3d.add_volume(grid, mapper="gpu", cmap="gray", opacity=opacities, reset_camera=False, show_scalar_bar=False)

    def normalize_data(self, data):
            # Step 1: Create a mask with value = 1 where data is not approximately equal to 0
            mask_data = np.where(np.abs(data) > 1e-3, 1, 0)
        
            # Step 2: Fill any holes in the mask
            mask_data = binary_fill_holes(mask_data).astype(int)
        
            # Step 3: Normalize the data
            brain_values = data[mask_data > 0]
            mean = np.mean(brain_values)
            stddev = np.std(brain_values)
        
            # Normalize data to zero-mean and unit variance
            normalized_data = np.where(mask_data > 0, (data - mean) / stddev, 0)
        
            return normalized_data

    def biquadratic(self, data, steepness=20, exponent=4, opacity_multiplier=200):
            min_val = data.min()
            max_val = data.max()
            # Create an array of data values (intensity levels)
            levels = np.linspace(min_val, max_val, 256)  # 256 levels is typical but can be adjusted
        
            # Determine the midpoint (closest to zero)
            mid_val = 0 if min_val <= 0 <= max_val else min_val if abs(min_val) < abs(max_val) else max_val

            # Normalize levels such that mid_val corresponds to the lowest point in the valley
            # Mapping the range [min_val, max_val] to [-1, 1] with mid_val as 0
            normalized_levels = 2 * (levels - mid_val) / (max_val - min_val)

            # Quadratic function creating a valley at mid_val with adjustable steepness
            # The function will have its minimum (most transparent point) at mid_val
            opacities = steepness * (normalized_levels ** exponent)  

            # Ensure opacities are within [0, 1]
            # Normalize again if needed, especially if adjusting contrast or depth of valley
            opacities = np.clip(opacities, 0, 1) * opacity_multiplier

            return opacities

    def add_plane_toggle_action(self, menu, name, key):
            action = QAction(name, self, checkable=True, checked=False)
            action.triggered.connect(lambda checked, k=key, a=action: self.handle_plane_visibility_toggle(k, a))
            menu.addAction(action)

    def handle_plane_visibility_toggle(self, plane_key, action):
            current_state = self.plane_visibility[plane_key]
            new_state = not current_state
            self.plane_visibility[plane_key] = new_state
            action.setChecked(new_state)
            self.update_planes()

    def toggle_plane_visibility(self, plane_key, visible):
            self.plane_visibility[plane_key] = visible
            self.update_planes()

    def update_planes(self):
            if self.data is None:
                self.clear_3d_planes()
                return
            dims = self.data.shape
            centers = {
                'axial': [dims[0] / 2, dims[1] / 2, self.axial_index],
                'coronal': [dims[0] / 2, self.coronal_index, dims[2] / 2],
                'sagittal': [self.sagittal_index, dims[1] / 2, dims[2] / 2]
            }
            directions = {
                'axial': (0, 0, 1),
                'coronal': (0, 1, 0),
                'sagittal': (1, 0, 0)
            }
            colors = {
                'axial': 'red',
                'coronal': 'green',
                'sagittal': 'yellow'
            }

            for key in ['axial', 'coronal', 'sagittal']:
                if self.plane_visibility[key]:
                    if key not in self.plane_meshes:
                        # Create the plane mesh once
                        plane = Plane(center=centers[key], direction=directions[key], i_size=dims[0], j_size=dims[1])
                        mesh = self.plotter_3d.add_mesh(plane, color=colors[key], opacity=0.6, reset_camera=False, name=f"{key}_plane")
                        self.plane_meshes[key] = (plane, mesh)
                    else:
                        # Update the position of the existing plane
                        plane, mesh = self.plane_meshes[key]
                        plane.origin = centers[key]
                        plane.update()
                        mesh.SetPosition(plane.center)
                else:
                    if key in self.plane_meshes:
                        # Remove the plane from the plotter
                        self.plotter_3d.remove_actor(self.plane_meshes[key][1])
                        del self.plane_meshes[key]

    def open_parameter_dialog(self):
            dialog = ParameterDialog(self)
            dialog.exec_()

    def update_visualization(self, reset_volume = False):
            # Redraw or recalculate the 3D visualization with the new parameters
            if self.data is None:
                self.plotter_3d.clear()
                self.clear_3d_planes()
                return
            if reset_volume:
                self.create_3d_volume(self.data, self.voxel_dims)
            self.update_3d_view()

    def get_orientation_by_view(self, view):
            for orientation, v in self.views.items():
                if v == view:
                    return orientation

    def get_aspect_ratio(self, voxel_dims, orientation):
            if orientation == 'axial':
                return voxel_dims[1] / voxel_dims[0]
            elif orientation == 'coronal':
                return voxel_dims[2] / voxel_dims[0]
            elif orientation == 'sagittal':
                return voxel_dims[2] / voxel_dims[1]
            else:
                return 1.0

    def highlight_voxel(self, coord, orientation, actor_id="mouse_hover"):
            # Determine the geometry based on the current segmentation tool
            if self.current_segmentation_tool in ("2d_brush", "3d_patch_contour"):
                # Cylinder parameters
                diameter = self.brush_size_slider.value()
                height = self.brush_thickness_slider.value()
                radius = diameter / 2.0
                if orientation == 'axial':
                    direction = (0, 0, 1)
                elif orientation == 'coronal':
                    direction = (0, 1, 0)
                elif orientation == 'sagittal':
                    direction = (1, 0, 0)

                # Adjusting coordinate to match selected brush position
                if self.brush_position == "Centered":
                    pass  # Default behavior
                elif self.brush_position == "Above":
                    coord = list(coord)
                    if orientation == 'axial':
                        coord[2] += height / 2
                    elif orientation == 'coronal':
                        coord[1] += height / 2
                    elif orientation == 'sagittal':
                        coord[0] += height / 2
                elif self.brush_position == "Below":
                    coord = list(coord)
                    if orientation == 'axial':
                        coord[2] -= height / 2
                    elif orientation == 'coronal':
                        coord[1] -= height / 2
                    elif orientation == 'sagittal':
                        coord[0] -= height / 2

                # Create a cylinder at the 3D coordinate
                cylinder = pv.Cylinder(center=coord, radius=radius, height=height, direction=direction)
                self.plotter_3d.add_mesh(cylinder, color='blue', render=True, reset_camera=False, name=actor_id)
            else:
                # Default to using a sphere for other tools
                sphere = pv.Sphere(radius=3, center=coord)
                self.plotter_3d.add_mesh(sphere, color='red', render=True, reset_camera=False, name=actor_id)

    def remove_mouse_highlights(self):
            if 'brush_circle' in self.highlight_actors:
                # Remove the circle from whichever scene actually holds it. The mouse may already
                # be in a different view than the one the circle was last drawn in.
                brush_item = self.highlight_actors.pop('brush_circle')
                try:
                    brush_scene = brush_item.scene()
                    if brush_scene is not None:
                        brush_scene.removeItem(brush_item)
                except RuntimeError:
                    pass  # underlying Qt item was already deleted (e.g. scene.clear())

            if 'mouse_hover' in self.highlight_actors:
                self.plotter_3d.remove_actor(self.highlight_actors['mouse_hover'])
                del self.highlight_actors['mouse_hover']

    def convert_slice_to_image(self, slice_data):
            # Normalize slice_data to 0-255 based on the global min and max
            if self.global_max > self.global_min:  # Avoid division by zero
                slice_data = (slice_data - self.global_min) / (self.global_max - self.global_min) * 255
            else:
                slice_data = np.zeros(slice_data.shape)  # Fallback in case of no range

            slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)

            height, width = slice_data.shape
            bytes_per_line = width  # Since each pixel is exactly one byte

            # Ensure the numpy array is contiguous, required by QImage
            slice_data = np.ascontiguousarray(slice_data)

            # Create QImage from numpy data
            image = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return image

    def assign_colors_to_levels(self, levels):
            # Reset or clear the previous level color map
            self.level_color_map = {}
            color_cycle = itertools.cycle(self.color_map)  # Create a repeating iterator over the colors
            for level in levels:
                if level != 0:  # Avoid assigning a color to the background
                    self.level_color_map[level] = next(color_cycle)

    def get_color_for_level(self, level, alpha):
            if level in self.invisible_levels:
                return (0, 0, 0, 0)  # Return RGBA tuple for transparent
            else:
                color_name = self.level_color_map.get(level, "white")
                rgb = mcolors.to_rgb(color_name)  # Convert color name to RGB tuple
                rgba = (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255, alpha * 255)  # Convert to RGBA tuple
                return rgba

    def display_color_key_on_3d(self):
            if self.data is None:
                return
            unique_levels = self.unique_levels
            if 0 in unique_levels:
                unique_levels = unique_levels[unique_levels != 0]  # Exclude the background level

            # Prepare a list of tuples for the legend, where each tuple is (label, color)
            level_names = self.segmentation_names.get(self.current_segmentation_key, {})
            legend_entries = [(level_names.get(level, f"Level {level}"), self.level_color_map.get(level, "white")) for level in unique_levels]

            # Clear any existing legend first
            self.plotter_3d.remove_legend()

            # Add the legend to the plotter
            self.plotter_3d.add_legend(labels=legend_entries, bcolor=None, size=(0.2, 1), border=False, loc='center left')

    def create_colormap(self, unique_levels, alpha=0.7, background_level=0):
            # Prepare a color list for the colormap
            colors = []
            for level in unique_levels:
                if level == background_level:
                    # Add transparent color for background
                    colors.append((0, 0, 0, 0))  # RGBA for transparency
                else:
                    colors.append = self.get_color_for_level(level, alpha)


            return ListedColormap(colors), Normalize(vmin=np.min(unique_levels), vmax=np.max(unique_levels))

    def update_3d_view(self):
            if self.data is None:
                return
            if self.current_segmentation is None:
                self.plotter_3d.remove_actor("segmentation_3d_highlight")
                self.plotter_3d.remove_legend()
                return

            # Generate a color map for the segmentation levels
            unique_levels = self.unique_levels
            visible_nonzero_levels = [int(level) for level in unique_levels if int(level) != 0 and int(level) not in self.invisible_levels]
            if len(visible_nonzero_levels) == 0:
                self.plotter_3d.remove_actor("segmentation_3d_highlight")
                self.plotter_3d.remove_legend()
                return

            # Assign colors to current levels
            self.assign_colors_to_levels(unique_levels)
            n_colors = np.max(unique_levels) + 1  # Include zero
            color_list = np.zeros((n_colors, 4), dtype=np.uint8)  # Include RGBA channels

            for level in range(n_colors):
                if level == 0:
                    color_list[level] = [0, 0, 0, 0]  # Fully transparent for background
                else:
                    color_list[level] = self.get_color_for_level(level, 0.4)

            # Adjust color brightness based on y-coordinate
            dims = self.current_segmentation.shape
            y_dim = dims[1]  # Assuming y is the second dimension
            brightness_factor = np.linspace(1, 0.7, y_dim)[:, None]  # Linearly interpolate from 1 to 0.5

            # Prepare for applying the brightness adjustment
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(self.current_segmentation.shape)
            flat_data_array = self.current_segmentation.flatten(order='F')
            vtk_array = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_data.GetPointData().SetScalars(vtk_array)
            grid = pv.ImageData(vtk_data)

            # Apply brightness adjustment to colors based on y-coordinate
            scalars = np.zeros((flat_data_array.size, 4), dtype=np.uint8)
            for level in range(n_colors):
                if level != 0:
                    indices = flat_data_array == level
                    y_coords = np.unravel_index(np.where(indices), dims)[1]  # Get y-coordinates
                    adjusted_colors = color_list[level, :3] * brightness_factor[y_coords]  # Adjust color brightness
                    scalars[indices, :3] = adjusted_colors
                    scalars[indices, 3] = color_list[level, 3]  # Set opacity

            # Add the new volume with adjusted colors
            self.plotter_3d.add_volume(grid, mapper="gpu", scalars=scalars, reset_camera=False, name="segmentation_3d_highlight", show_scalar_bar=False)
            self.display_color_key_on_3d()

    def convert_mask_to_color_mask(self, mask, alpha=0.25):
            unique_levels = self.unique_levels
            self.assign_colors_to_levels(unique_levels)
            M, N = mask.shape
            color_mask = np.zeros((M, N, 4), dtype=np.uint8)  # Default to transparent
            for level in unique_levels:
                if level != 0:  # Assuming '0' is the background/non-segmented level
                    rgba_color = self.get_color_for_level(level, alpha)
                    color_mask[mask == level] = rgba_color
            return color_mask

    def setup_2d_viewer_borders(self):
            self.view_axial.setStyleSheet("border: 2px solid red;")
            self.view_coronal.setStyleSheet("border: 2px solid green;")
            self.view_sagittal.setStyleSheet("border: 2px solid yellow;")

    def display_all_slices(self):
            # First, update the planes' positions according to the current slice indices
            self.update_planes()

            # Display each slice in its respective viewer
            self.display_axial_slice()
            self.display_coronal_slice()
            self.display_sagittal_slice()

    def get_roi_slice_2d(self, orientation):
            if self.roi_mask is None:
                return None

            if orientation == 'axial':
                roi_slice = self.roi_mask[:, :, self.axial_index]
            elif orientation == 'coronal':
                roi_slice = self.roi_mask[:, self.coronal_index, :]
            elif orientation == 'sagittal':
                roi_slice = self.roi_mask[self.sagittal_index, :, :]
            else:
                roi_slice = None
                print(f"Unknown orientation: {orientation}")

            if roi_slice is not None and roi_slice.dtype != bool:
                roi_slice = roi_slice.astype(bool)

            return roi_slice

    def update_2d_brush_highlights(self, orientation):
            if self.current_segmentation is None:
                return

            # Initialize the merged_segmentation with the current segmentation slice cache
            merged_segmentation = np.copy(self.current_seg_slice_cache_2d)
        
            roi_slice = self.get_roi_slice_2d(orientation)
            if roi_slice is not None:
                merged_brush_cache = roi_slice & (self.brush_cache_2d != 0)
            else:
                merged_brush_cache = self.brush_cache_2d

            if self.new_segmentation_overlap == "add_around":
                # Only update where current segmentation is zero
                mask = (self.current_seg_slice_cache_2d == 0) & (merged_brush_cache == 1)
                merged_segmentation[mask] = self.segmentation_level

            elif self.new_segmentation_overlap == "overwrite":
                # Overwrite anywhere the brush has marked
                mask = (merged_brush_cache == 1) & np.isin(self.current_seg_slice_cache_2d, list(self.overwrite_levels))
                merged_segmentation[mask] = self.segmentation_level

            elif self.new_segmentation_overlap == "erase":
                # Set to 0 anywhere the brush has marked
                mask = (merged_brush_cache == 1)
                merged_segmentation[mask] = 0

            elif self.new_segmentation_overlap == "erase_level":
                # Erase only the current level where brush is marked
                mask = (self.current_seg_slice_cache_2d == self.segmentation_level) & (merged_brush_cache == 1)
                merged_segmentation[mask] = 0

            elif self.new_segmentation_overlap == "overlap_only":
                overlap_level = self.overlap_level_combo.currentText()
                if overlap_level != "All":
                    mask = (self.current_seg_slice_cache_2d == int(overlap_level)) & (merged_brush_cache == 1)
                    merged_segmentation[mask] = self.segmentation_level
                else:
                    mask = (self.current_seg_slice_cache_2d != 0) & (merged_brush_cache == 1)
                    merged_segmentation[mask] = self.segmentation_level

            # Update the QGraphicsView for each orientation
            self.display_slice(self.brush_slice_cache, merged_segmentation, self.views[orientation], self.voxel_dims, orientation)

    def display_slice(self, slice_data, mask_slice, view, voxel_dims, orientation):
            # Rotate the slice data 90 degrees counter-clockwise for correct orientation
            slice_data = np.rot90(slice_data)
            if mask_slice is not None:
                mask_slice = np.rot90(mask_slice)
            # Proceed with converting to QImage and the rest of your code
            image = self.convert_slice_to_image(slice_data)
            pixmap = QPixmap.fromImage(image)

            if mask_slice is not None:
                # Create a color overlay for the mask
                color_mask = self.convert_mask_to_color_mask(mask_slice)

                # Convert color_mask to QImage
                color_mask = np.ascontiguousarray(color_mask)
                height, width, channels = color_mask.shape
                bytes_per_line = channels * width
                mask_image = QImage(color_mask.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
                mask_pixmap = QPixmap.fromImage(mask_image)

                # Blend the mask onto the original image using QPainter
                painter = QPainter(pixmap)
                painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                painter.drawPixmap(0, 0, mask_pixmap)
                painter.end()

            # Update the pixmap in the view
            existing_item = getattr(view, 'pixmap_item', None)
            existing_scene = view.scene()
            if existing_item is None or existing_scene is None or existing_item.scene() is not existing_scene:
                scene = QGraphicsScene()
                pixmap_item = scene.addPixmap(pixmap)
                view.setScene(scene)
                view.pixmap_item = pixmap_item
            else:
                view.pixmap_item.setPixmap(pixmap)

           # Instead of resetting the transform, adjust only if necessary
            aspect_ratio = self.get_aspect_ratio(voxel_dims, orientation)
            if not hasattr(view, 'aspect_ratio') or view.aspect_ratio != aspect_ratio:
                # Calculate the scale factor to adjust aspect ratio
                scale_x = 1
                scale_y = aspect_ratio
                if hasattr(view, 'current_scale'):
                    # Adjust for any existing scaling
                    scale_x /= view.current_scale[0]
                    scale_y /= view.current_scale[1]
                view.scale(scale_x, scale_y)
                view.aspect_ratio = aspect_ratio
                view.current_scale = (scale_x, scale_y)

            view.setSceneRect(view.scene().itemsBoundingRect())

    def display_axial_slice(self):
            if self.data is None:
                return
            data_slice = self.get_current_slice_data('axial')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[:, :, self.axial_index]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_axial, self.voxel_dims, 'axial')

    def display_coronal_slice(self):
            if self.data is None:
                return
            data_slice = self.get_current_slice_data('coronal')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[:, self.coronal_index, :]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_coronal, self.voxel_dims, 'coronal')

    def display_sagittal_slice(self):
            if self.data is None:
                return
            data_slice = self.get_current_slice_data('sagittal')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[self.sagittal_index, :, :]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_sagittal, self.voxel_dims, 'sagittal')
