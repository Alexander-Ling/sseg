"""Rendering helpers and view update mixins for the sseg viewer."""

import itertools
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import ListedColormap, Normalize
from pyvista import Plane
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QAction, QGraphicsEllipseItem, QGraphicsScene
from vtk.util import numpy_support
from scipy.ndimage import binary_fill_holes

from . import perf
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

    def get_3d_downsample_factor(self):
            """Linear downsampling used for the 3D view only (1 = full resolution, 2 = 'Fast' mode)."""
            return 2 if getattr(self, 'quality_3d', 'full') == 'fast' else 1

    @staticmethod
    def _block_reduce(volume, factor, reducer):
            """Reduce factor^3 blocks with `reducer` ('mean' or 'max'). Trailing voxels that do not fill a block are dropped."""
            if factor == 1:
                return volume
            nx, ny, nz = (n // factor for n in volume.shape)
            cropped = volume[:nx * factor, :ny * factor, :nz * factor]
            blocks = cropped.reshape(nx, factor, ny, factor, nz, factor)
            if reducer == 'max':
                return blocks.max(axis=(1, 3, 5))
            return blocks.mean(axis=(1, 3, 5), dtype=np.float32)

    @staticmethod
    def _grid_geometry_3d(factor):
            """Spacing/origin that keep the downsampled grid in original voxel-index coordinates (block centres)."""
            return (float(factor),) * 3, ((factor - 1) / 2.0,) * 3

    def _invalidate_3d_state(self):
            self._seg3d = None
            self._vol3d = None
            self._hover_state = None
            self.plane_meshes.clear()

    def create_3d_volume(self, data, voxel_dims):
            if data is None:
                self.plotter_3d.clear()
                self._invalidate_3d_state()
                self.clear_3d_planes()
                return
            if not getattr(self, 'show_3d_view', True):
                return
            factor = self.get_3d_downsample_factor()
            if factor > 1:
                data = self._block_reduce(data, factor, 'mean')
            # Create a vtkImageData object
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(data.shape)
            spacing, origin = self._grid_geometry_3d(factor)
            vtk_data.SetSpacing(spacing)
            vtk_data.SetOrigin(origin)

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
            self._invalidate_3d_state()

            # Auto 3D opacity is currently deprecated; keep the fallback path
            # available for backwards compatibility, but default to manual.
            value_range = (float(data.min()), float(data.max()))
            if getattr(self, "auto_3d_opacity", False):
                opacities = self.compute_auto_opacity_transfer_function(data)
            else:
                opacities = self.biquadratic(data, self.steepness, self.exponent, self.opacity_multiplier, value_range=value_range)

            if self.new_camera:
                actor = self._add_volume(grid, cmap="gray", opacity=opacities, show_scalar_bar=False)
                self.new_camera = False
            else:
                actor = self._add_volume(grid, cmap="gray", opacity=opacities, reset_camera=False, show_scalar_bar=False)
            # plotter.mapper is the mapper just created by add_volume; keep its lookup table for in-place opacity edits
            self._vol3d = {'actor': actor, 'range': value_range, 'factor': factor, 'lut': self.plotter_3d.mapper.lookup_table}

    def update_3d_opacity(self):
            """Apply new opacity settings to the existing 3D volume without re-uploading it to the GPU."""
            state = getattr(self, '_vol3d', None)
            valid = (
                state is not None
                and getattr(self, 'show_3d_view', True)
                and any(a is state['actor'] for a in self.plotter_3d.renderer.actors.values())
            )
            if not valid or getattr(self, 'auto_3d_opacity', False):
                self.create_3d_volume(self.data, self.voxel_dims)
                return
            actor = state['actor']
            opacities = self.biquadratic(None, self.steepness, self.exponent, self.opacity_multiplier, value_range=state['range'])
            lut = state['lut']
            lut.apply_opacity(opacities)
            actor.GetProperty().SetScalarOpacity(lut.to_opacity_tf())
            self.request_3d_render()

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

    def biquadratic(self, data, steepness=20, exponent=4, opacity_multiplier=200, value_range=None):
            min_val, max_val = value_range if value_range is not None else (data.min(), data.max())
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
            if not getattr(self, 'show_3d_view', True):
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

    def update_visualization(self, reset_volume=False, dirty_bbox=None):
            """Redraw the 3D view. `dirty_bbox` ((x0,x1),(y0,y1),(z0,z1)) limits the segmentation overlay refresh
            to the edited region; None means everything may have changed."""
            if self.data is None:
                self.plotter_3d.clear()
                self._invalidate_3d_state()
                self.clear_3d_planes()
                return
            if not getattr(self, 'show_3d_view', True):
                return
            if reset_volume:
                self.create_3d_volume(self.data, self.voxel_dims)
                self.update_planes()  # plotter.clear() removed the plane actors; re-add any that are enabled
                self.update_3d_view()
            else:
                self.schedule_3d_view_update(dirty_bbox)

    def schedule_3d_view_update(self, dirty_bbox=None):
            """Refresh the 3D segmentation overlay ~250 ms after the last edit instead of after every one."""
            pending = getattr(self, '_seg3d_pending', None)
            if dirty_bbox is None or pending == 'full':
                self._seg3d_pending = 'full'
            else:
                self._seg3d_pending = self.union_bbox(pending, dirty_bbox)
            timer = getattr(self, '_seg3d_timer', None)
            if timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(250)
                timer.timeout.connect(self.update_3d_view)
                self._seg3d_timer = timer
            timer.start()  # restarts the countdown if already running

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

    def request_3d_render(self):
            """Render the 3D view at most once per ~33 ms, however many callers ask."""
            if not getattr(self, 'show_3d_view', True):
                return
            timer = getattr(self, '_render_timer', None)
            if timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(33)
                timer.timeout.connect(self._do_3d_render)
                self._render_timer = timer
            if not timer.isActive():
                timer.start()

    def _do_3d_render(self):
            try:
                self.plotter_3d.render()
            except Exception:
                pass

    def _hover_marker_actor(self, key, build_mesh, color):
            """Return the cached 3D hover marker actor for `key`, creating it once (or again after plotter.clear())."""
            name = f"mouse_hover_{key}"
            actor = self.plotter_3d.renderer.actors.get(name)
            if actor is None:
                actor = self.plotter_3d.add_mesh(build_mesh(), color=color, render=False, reset_camera=False, name=name)
                self._hover_marker_names = getattr(self, '_hover_marker_names', set()) | {name}
            return name, actor

    def hide_hover_markers(self):
            self._hover_state = None
            names = getattr(self, '_hover_marker_names', None)
            plotter = getattr(self, 'plotter_3d', None)
            if not names or plotter is None:
                return
            changed = False
            actors = plotter.renderer.actors
            for name in getattr(self, '_hover_marker_names', ()):
                actor = actors.get(name)
                if actor is not None and actor.GetVisibility():
                    actor.SetVisibility(False)
                    changed = True
            if changed:
                self.request_3d_render()

    @perf.timed("highlight_voxel")
    def highlight_voxel(self, coord, orientation, actor_id="mouse_hover"):
            # The marker is a single cached actor that is moved, not rebuilt, and renders are throttled.
            if not getattr(self, 'show_3d_hover_marker', True) or not getattr(self, 'show_3d_view', True):
                return
            # Determine the geometry based on the current segmentation tool
            if self.current_segmentation_tool in ("2d_brush", "3d_patch_contour"):
                # Cylinder parameters
                diameter = self.brush_size_slider.value()
                height = self.brush_thickness_slider.value()
                radius = diameter / 2.0
                if orientation == 'axial':
                    direction = (0, 0, 1)
                    scale = (radius, radius, height)
                elif orientation == 'coronal':
                    direction = (0, 1, 0)
                    scale = (radius, height, radius)
                elif orientation == 'sagittal':
                    direction = (1, 0, 0)
                    scale = (height, radius, radius)

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

                key = f"cylinder_{orientation}"
                build = lambda: pv.Cylinder(center=(0, 0, 0), radius=1.0, height=1.0, direction=direction)
                color = 'blue'
                position = tuple(float(c) for c in coord)
            else:
                # Default to using a sphere for other tools
                key = "sphere"
                build = lambda: pv.Sphere(radius=3, center=(0, 0, 0))
                color = 'red'
                scale = (1.0, 1.0, 1.0)
                position = tuple(float(c) for c in coord)

            state = (key, position, scale)
            if state == getattr(self, '_hover_state', None):
                return
            name, actor = self._hover_marker_actor(key, build, color)
            for other in getattr(self, '_hover_marker_names', ()):
                if other != name:
                    other_actor = self.plotter_3d.renderer.actors.get(other)
                    if other_actor is not None:
                        other_actor.SetVisibility(False)
            actor.SetScale(*scale)
            actor.SetPosition(*position)
            actor.SetVisibility(True)
            self._hover_state = state
            self.request_3d_render()

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

            self.hide_hover_markers()

    def _slice_to_uint8(self, slice_data):
            # Normalize slice_data to 0-255 based on the global min and max
            if self.global_max > self.global_min:  # Avoid division by zero
                slice_data = (slice_data - self.global_min) / (self.global_max - self.global_min) * 255
            else:
                slice_data = np.zeros(slice_data.shape)  # Fallback in case of no range

            return np.ascontiguousarray(np.clip(slice_data, 0, 255).astype(np.uint8))

    def convert_slice_to_image(self, slice_data):
            slice_data = self._slice_to_uint8(slice_data)

            height, width = slice_data.shape
            bytes_per_line = width  # Since each pixel is exactly one byte

            # Create QImage from numpy data
            image = QImage(slice_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            return image.copy()  # detach from the numpy buffer, which is freed when this function returns

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

    @perf.timed("update_3d_view")
    def update_3d_view(self):
            pending = getattr(self, '_seg3d_pending', None) or 'full'
            self._seg3d_pending = None
            if self.data is None or not getattr(self, 'show_3d_view', True):
                return
            if self.current_segmentation is None:
                self.plotter_3d.remove_actor("segmentation_3d_highlight")
                self.plotter_3d.remove_legend()
                self._seg3d = None
                return

            # Generate a color map for the segmentation levels
            unique_levels = self.unique_levels
            visible_nonzero_levels = [int(level) for level in unique_levels if int(level) != 0 and int(level) not in self.invisible_levels]
            if len(visible_nonzero_levels) == 0:
                self.plotter_3d.remove_actor("segmentation_3d_highlight")
                self.plotter_3d.remove_legend()
                self._seg3d = None
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

            factor = self.get_3d_downsample_factor()
            dims = tuple(n // factor for n in self.current_segmentation.shape)
            brightness_factor = np.linspace(1, 0.7, dims[1])  # Linearly interpolate from 1 to 0.7 along y
            color_key = (tuple(int(l) for l in unique_levels), frozenset(int(l) for l in self.invisible_levels),
                         tuple(self.color_map), factor, dims)

            state = getattr(self, '_seg3d', None)
            reusable = (
                state is not None
                and state['dims'] == dims
                and state['factor'] == factor
                and any(a is state['actor'] for a in self.plotter_3d.renderer.actors.values())
            )

            def colorize(volume, origin_voxel):
                """RGBA (uint8) for a block of the (possibly pooled) labels. `origin_voxel` is the block's corner in the
                full 3D grid. The brightness ramp's y index is derived with C-order arithmetic on the F-order flat
                index, exactly as the original implementation did."""
                rgba = np.zeros(volume.shape + (4,), dtype=np.uint8)
                ix, iy, iz = np.nonzero(volume)
                levels_at = volume[ix, iy, iz].astype(np.int64)
                keep = (levels_at > 0) & (levels_at < n_colors)
                if not keep.all():
                    ix, iy, iz, levels_at = ix[keep], iy[keep], iz[keep], levels_at[keep]
                flat_index = (ix + origin_voxel[0]) + dims[0] * ((iy + origin_voxel[1]) + dims[1] * (iz + origin_voxel[2]))
                y_coords = (flat_index // dims[2]) % dims[1]
                rgba[ix, iy, iz, :3] = (color_list[levels_at, :3] * brightness_factor[y_coords][:, None]).astype(np.uint8)
                rgba[ix, iy, iz, 3] = color_list[levels_at, 3]
                return rgba

            if reusable:
                dataset = state['actor'].mapper.dataset
                buffer = dataset.point_data[state['name']]
                # The buffer is indexed by F-order flat voxel index; this view addresses it as [z, y, x, rgba].
                buffer4 = buffer.reshape(dims[2], dims[1], dims[0], 4)
                if pending != 'full' and state['color_key'] == color_key:
                    # Partial update: only the edited region (in pooled coordinates) is recomputed and written.
                    lo = [bounds[0] // factor for bounds in pending]
                    hi = [min(-(-bounds[1] // factor), n) for bounds, n in zip(pending, dims)]
                    sub = self.current_segmentation[tuple(slice(l * factor, h * factor) for l, h in zip(lo, hi))]
                    rgba = colorize(self._block_reduce(sub, factor, 'max'), lo)
                    buffer4[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]] = rgba.transpose(2, 1, 0, 3)
                else:
                    # Everything may have changed (levels, colours or visibility): rewrite the whole buffer in place.
                    seg = self._block_reduce(self.current_segmentation, factor, 'max')
                    buffer4[:] = colorize(seg, (0, 0, 0)).transpose(2, 1, 0, 3)
                    state['color_key'] = color_key
                    if state['legend_key'] != self._legend_key():
                        self.display_color_key_on_3d()
                        state['legend_key'] = self._legend_key()
                buffer.VTKObject.Modified()
                dataset.Modified()
                self.request_3d_render()
                return

            # No usable volume yet: build it
            seg = self._block_reduce(self.current_segmentation, factor, 'max')
            scalars = np.ascontiguousarray(colorize(seg, (0, 0, 0)).transpose(2, 1, 0, 3)).reshape(-1, 4)
            spacing, origin = self._grid_geometry_3d(factor)
            grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

            # Add the new volume with adjusted colors
            actor = self._add_volume(grid, scalars=scalars, reset_camera=False, name="segmentation_3d_highlight", show_scalar_bar=False)
            self.display_color_key_on_3d()
            self._seg3d = {
                'actor': actor, 'dims': dims, 'factor': factor, 'name': 'Data',
                'color_key': color_key, 'legend_key': self._legend_key(),
            }

    def _add_volume(self, grid, **kwargs):
            """add_volume with the GPU ray-cast mapper, falling back to VTK's 'smart' mapper if that is unavailable.
            Set SSEG_VOLUME_MAPPER (e.g. smart, fixed_point) to force a different mapper on machines with a poor GPU."""
            preferred = os.environ.get("SSEG_VOLUME_MAPPER", "gpu")
            try:
                return self.plotter_3d.add_volume(grid, mapper=preferred, **kwargs)
            except Exception:
                if preferred == "smart":
                    raise
                return self.plotter_3d.add_volume(grid, mapper="smart", **kwargs)

    def _legend_key(self):
            names = self.segmentation_names.get(self.current_segmentation_key, {})
            return (tuple(int(l) for l in self.unique_levels), tuple(sorted(names.items())), tuple(sorted(self.level_color_map.items())))

    def _overlay_lut(self, alpha):
            """RGBA lookup table indexed by segmentation level; rebuilt only when levels, visibility or colours change."""
            levels = tuple(int(level) for level in self.unique_levels)
            key = (levels, frozenset(int(l) for l in self.invisible_levels), tuple(self.color_map), alpha)
            cached = getattr(self, '_lut_cache', None)
            if cached is not None and cached[0] == key:
                return cached[1]
            self.assign_colors_to_levels(self.unique_levels)
            lut = np.zeros((max(levels) + 1 if levels and min(levels) >= 0 else 1, 4), dtype=np.uint8)
            if levels and min(levels) >= 0:
                for level in levels:
                    if level != 0:  # '0' is the background/non-segmented level
                        lut[level] = self.get_color_for_level(level, alpha)
            self._lut_cache = (key, lut)
            return lut

    def convert_mask_to_color_mask(self, mask, alpha=0.25):
            lut = self._overlay_lut(alpha)
            mask = np.asarray(mask)
            if mask.dtype.kind not in 'iu':
                mask = mask.astype(np.int64)
            if mask.size and (mask.min() < 0 or mask.max() >= lut.shape[0]):
                # Levels outside the table stay transparent (same as levels missing from unique_levels).
                color_mask = np.zeros(mask.shape + (4,), dtype=np.uint8)
                valid = (mask >= 0) & (mask < lut.shape[0])
                color_mask[valid] = lut[mask[valid]]
                return color_mask
            return lut[mask]

    def _compose_overlay_rgb(self, gray, rgba):
            """Alpha-blend an RGBA overlay onto a uint8 grayscale slice; returns (H, W, 3) uint8, or None if nothing to draw."""
            alpha = rgba[..., 3]
            sel = alpha > 0
            if not sel.any():
                return None
            out = np.repeat(gray[..., None], 3, axis=2)
            a = alpha[sel].astype(np.uint16)[:, None]
            src = rgba[sel][:, :3].astype(np.uint16)
            dst = out[sel].astype(np.uint16)
            out[sel] = ((src * a + dst * (255 - a) + 127) // 255).astype(np.uint8)
            return out

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

    def _in_overwrite_levels(self, slice_data):
            """Boolean mask of voxels whose level is one of overwrite_levels (lookup table cached per level set)."""
            levels = frozenset(int(l) for l in self.overwrite_levels)
            cached = getattr(self, '_overwrite_lut', None)
            if cached is None or cached[0] != levels:
                nonneg = [l for l in levels if l >= 0]
                lut = np.zeros(max(nonneg) + 1 if nonneg else 1, dtype=bool)
                lut[nonneg] = True
                cached = (levels, lut)
                self._overwrite_lut = cached
            lut = cached[1]
            in_range = (slice_data >= 0) & (slice_data < lut.size)
            out = np.zeros(slice_data.shape, dtype=bool)
            out[in_range] = lut[slice_data[in_range]]
            return out

    def update_2d_brush_highlights(self, orientation):
            if self.current_segmentation is None or not self.view_enabled.get(orientation, True):
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
                mask = (merged_brush_cache == 1) & self._in_overwrite_levels(self.current_seg_slice_cache_2d)
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

    @perf.timed("display_slice")
    def display_slice(self, slice_data, mask_slice, view, voxel_dims, orientation):
            # Rotate the slice data 90 degrees counter-clockwise for correct orientation
            slice_data = np.rot90(slice_data)
            if mask_slice is not None:
                mask_slice = np.rot90(mask_slice)
            image = None
            composed = None
            if mask_slice is not None:
                # Colour the mask via the cached lookup table and blend it into the slice in numpy
                gray = self._slice_to_uint8(slice_data)
                composed = self._compose_overlay_rgb(gray, self.convert_mask_to_color_mask(mask_slice))
                if composed is not None:
                    composed = np.ascontiguousarray(composed)
                    height, width, _ = composed.shape
                    image = QImage(composed.data, width, height, 3 * width, QImage.Format_RGB888)
            if image is None:
                image = self.convert_slice_to_image(slice_data)
            pixmap = QPixmap.fromImage(image)  # copies the pixel data, so the numpy buffer may be released afterwards

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

            # Only recompute the scene rect when the image size changed (it is expensive and the rect is otherwise constant)
            size_key = (pixmap.width(), pixmap.height(), view.scene())
            if getattr(view, '_scene_rect_key', None) != size_key:
                view.setSceneRect(view.scene().itemsBoundingRect())
                view._scene_rect_key = size_key

    def display_axial_slice(self):
            if self.data is None or not self.view_enabled.get('axial', True):
                return
            data_slice = self.get_current_slice_data('axial')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[:, :, self.axial_index]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_axial, self.voxel_dims, 'axial')

    def display_coronal_slice(self):
            if self.data is None or not self.view_enabled.get('coronal', True):
                return
            data_slice = self.get_current_slice_data('coronal')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[:, self.coronal_index, :]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_coronal, self.voxel_dims, 'coronal')

    def display_sagittal_slice(self):
            if self.data is None or not self.view_enabled.get('sagittal', True):
                return
            data_slice = self.get_current_slice_data('sagittal')
            if not self.current_segmentation is None:
                mask_slice = self.current_segmentation[self.sagittal_index, :, :]
            else:
                mask_slice = None
            self.display_slice(data_slice, mask_slice, self.view_sagittal, self.voxel_dims, 'sagittal')
