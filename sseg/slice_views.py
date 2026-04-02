"""2D slice interaction and event-handling mixins for the sseg viewer."""

import numpy as np
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QColor, QCursor, QMouseEvent, QPen, QTransform
from PyQt5.QtWidgets import QApplication, QGraphicsEllipseItem, QGraphicsPixmapItem


class SliceViewMixin:
    def get_current_slice_data(self, orientation):
            if self.data is None or None in (self.axial_index, self.coronal_index, self.sagittal_index):
                return None
            if orientation == 'axial':
                return self.data[:, :, self.axial_index]
            elif orientation == 'coronal':
                return self.data[:, self.coronal_index, :]
            elif orientation == 'sagittal':
                return self.data[self.sagittal_index, :, :]

    def get_current_segmentation_slice_data(self, orientation):
            if self.current_segmentation is None:
                return None
            if not np.all(self.current_segmentation == 0):
                if orientation == 'axial':
                    return self.current_segmentation[:, :, self.axial_index]
                elif orientation == 'coronal':
                    return self.current_segmentation[:, self.coronal_index, :]
                elif orientation == 'sagittal':
                    return self.current_segmentation[self.sagittal_index, :, :]
            return None

    def update_brush_circle(self, scene_pos, view):
            brush_size = self.brush_size_slider.value()  # Get the current brush size
            radius = brush_size / 2

            # Check if 'brush_circle' is in highlight_actors
            if 'brush_circle' in self.highlight_actors:
                # Attempt to check if the item is still valid and in the correct scene
                try:
                    if self.highlight_actors['brush_circle'].scene() != view.scene():
                        # If it belongs to a different scene, remove it safely
                        try:
                            self.highlight_actors['brush_circle'].scene().removeItem(self.highlight_actors['brush_circle'])
                        except RuntimeError:
                            # Handle cases where the item is already deleted
                            pass
                        del self.highlight_actors['brush_circle']

                        # Create a new QGraphicsEllipseItem for the current view
                        self.highlight_actors['brush_circle'] = QGraphicsEllipseItem()
                        self.highlight_actors['brush_circle'].setPen(QPen(QColor("blue"), 2, Qt.SolidLine))
                        view.scene().addItem(self.highlight_actors['brush_circle'])
                        self.highlight_actors['brush_circle'].setRect(scene_pos.x() - radius, scene_pos.y() - radius, brush_size, brush_size)
                        self.highlight_actors['brush_circle'].setVisible(True)
                    else:
                        # Update the existing brush_circle's position and size
                        self.highlight_actors['brush_circle'].setRect(scene_pos.x() - radius, scene_pos.y() - radius, brush_size, brush_size)
                        self.highlight_actors['brush_circle'].setVisible(True)
                except RuntimeError:
                    # Handle cases where the item is already deleted and recreate it
                    self.highlight_actors['brush_circle'] = QGraphicsEllipseItem()
                    self.highlight_actors['brush_circle'].setPen(QPen(QColor("blue"), 2, Qt.SolidLine))
                    view.scene().addItem(self.highlight_actors['brush_circle'])
                    self.highlight_actors['brush_circle'].setRect(scene_pos.x() - radius, scene_pos.y() - radius, brush_size, brush_size)
                    self.highlight_actors['brush_circle'].setVisible(True)
            else:
                # Create a new QGraphicsEllipseItem since it does not exist
                self.highlight_actors['brush_circle'] = QGraphicsEllipseItem()
                self.highlight_actors['brush_circle'].setPen(QPen(QColor("blue"), 2, Qt.SolidLine))
                view.scene().addItem(self.highlight_actors['brush_circle'])
                self.highlight_actors['brush_circle'].setRect(scene_pos.x() - radius, scene_pos.y() - radius, brush_size, brush_size)
                self.highlight_actors['brush_circle'].setVisible(True)

    def eventFilter(self, source, event):

            # Handling mouse button press
            if event.type() == QEvent.MouseButtonPress:
                if event.buttons() == Qt.RightButton:  # Use right mouse click for panning
                    self.panning = True
                    self.last_pan_point = event.pos()
                    source.setCursor(QCursor(Qt.ClosedHandCursor))
                    return True
                elif event.buttons() == Qt.LeftButton and source in [self.view_axial.viewport(), self.view_coronal.viewport(), self.view_sagittal.viewport()]:
                    if self.current_segmentation is not None and self.current_segmentation_tool:
                        if self.current_segmentation_tool in ("2d_brush", "3d_patch_contour") and self.coord:
                            self.is_drawing_2d_brush = True
                            self.init_2d_brush_cache()
                            self.apply_2d_brush(self.coord)
                            return True
                        elif self.current_segmentation_tool == "remove_volume" and self.coord:
                            self.apply_remove_contiguous_volume()
                            return True
                        elif self.current_segmentation_tool == "keep_volume_only" and self.coord:
                            self.apply_keep_only_contiguous_volume()
                            return True
                        elif self.current_segmentation_tool == "change_volume_level" and self.coord:
                            self.apply_change_volume_level()
                            return True
                        elif self.current_segmentation_tool == "fill_holes" and self.coord:
                            self.apply_fill_contiguous_volume_holes()
                            return True
                        elif self.current_segmentation_tool == "3d_contour" and self.coord:
                            intensity_percentage, gradient_percentage, max_ticks = self.get_gradient_parameters()
                            new_segmentation = self.gradient_tracing(seed_point=self.coord, base_value=self.pixel_value,
                                                                    intensity_percentage=intensity_percentage,
                                                                    gradient_percentage=gradient_percentage, max_ticks=max_ticks)
                            self.handle_segmentation_update(new_segmentation)
                            return True
                        elif self.current_segmentation_tool == "3d_contour_global" and self.coord:
                            intensity_percentage, gradient_percentage, max_ticks = self.get_gradient_parameters()
                            new_segmentation = self.gradient_tracing_global(seed_point=self.coord, base_value=self.pixel_value,
                                                                    intensity_percentage=intensity_percentage,
                                                                    gradient_percentage=gradient_percentage, max_ticks=max_ticks)
                            self.handle_segmentation_update(new_segmentation)
                            return True

            # Handling mouse button release
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.RightButton:
                    self.panning = False
                    source.unsetCursor()
                    return True
                elif event.button() == Qt.LeftButton and self.current_segmentation_tool == "2d_brush":
                    self.finalize_2d_brush_strokes()
                    self.is_drawing_2d_brush = False
                    return True
                elif event.button() == Qt.LeftButton and self.current_segmentation_tool == "3d_patch_contour":
                    ROI = self.finalize_2d_brush_strokes(returnROI=True)
                    self.is_drawing_2d_brush = False
                    self.execute_patch_based_region_growth(ROI)
                    return True

            # Handling mouse movement
            elif event.type() == QEvent.MouseMove:
                view = next((v for v in self.views.values() if v.viewport() is source), None)
                if view:
                    self.current_view = view
                    if self.panning:
                        delta = event.pos() - self.last_pan_point
                        view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() - delta.x())
                        view.verticalScrollBar().setValue(view.verticalScrollBar().value() - delta.y())
                        self.last_pan_point = event.pos()
                        return True
                    else:
                        self.track_pixel(event, view)
                        if self.current_segmentation_tool in ("2d_brush", "3d_patch_contour") and self.coord:
                            scene_pos = view.mapToScene(event.pos())
                            self.update_brush_circle(scene_pos, view)
                        if self.is_drawing_2d_brush and self.coord and self.current_segmentation_tool in ("2d_brush", "3d_patch_contour"):
                            self.apply_2d_brush(self.coord)
                            return True

            # Handling zoom and scroll
            elif event.type() == QEvent.Wheel:
                modifiers = QApplication.keyboardModifiers()
                delta = event.angleDelta().y()
                if modifiers == Qt.ControlModifier:
                    factor = 1.2 if delta > 0 else 0.8
                    for view in self.views.values():
                        if source == view.viewport():
                            self.zoom_view(view, factor, event.pos())
                    return True
                else:
                    scroll_speed = 5 if modifiers == Qt.ShiftModifier else 1
                    for orientation, view in self.views.items():
                        if source == view.viewport():
                            self.handle_scroll(orientation, delta, scroll_speed)
                            self.track_pixel(event, view)
                    return True

            return super().eventFilter(source, event)

    def track_pixel(self, event, view):
            if self.data is None or self.voxel_dims is None:
                self.pixel_value = None
                self.coord = None
                self.info_label.setText("No exam loaded")
                self.remove_mouse_highlights()
                return False
            #event.pos() is position relative to top left corner of view window. Changes as view window is resized.
            scene_pos = view.mapToScene(event.pos()) #Coordinates relative to static coordinate system. Plotted object remains in same location on these coordinates regardless of zoom, translation, etc.
            # Iterate through all items and find the QGraphicsPixmapItem
            pixmap_item = next((item for item in view.scene().items() if isinstance(item, QGraphicsPixmapItem)), None)
            if pixmap_item:
                # Convert the scene position directly to pixmap item coordinates
                item_pos = pixmap_item.mapFromScene(scene_pos)

                # Get the scale factors applied due to aspect ratio adjustments
                orientation = self.get_orientation_by_view(view)
                if orientation == 'axial':
                    aspect_ratio = self.voxel_dims[1] / self.voxel_dims[0]
                elif orientation == 'coronal':
                    aspect_ratio = self.voxel_dims[2] / self.voxel_dims[0]
                elif orientation == 'sagittal':
                    aspect_ratio = self.voxel_dims[2] / self.voxel_dims[1]

                if aspect_ratio > 1:
                    x_middle = (pixmap_item.pixmap().width()/2)
                    x_from_middle = item_pos.x() - x_middle
                    true_from_middle = x_from_middle * aspect_ratio
                    true_x = x_middle + true_from_middle
                    x, y = int(pixmap_item.pixmap().height() - item_pos.y()), int(true_x)
                elif aspect_ratio < 1:
                    y_middle = (pixmap_item.pixmap().height()/2)
                    y_from_middle = item_pos.y() - y_middle
                    true_from_middle = y_from_middle / aspect_ratio
                    true_y = y_middle + true_from_middle
                    x, y = int(pixmap_item.pixmap().height() - true_y), int(item_pos.x())
                else:
                    x, y = int(pixmap_item.pixmap().height() - item_pos.y()), int(item_pos.x())

                slice_data = self.get_current_slice_data(orientation)
                if slice_data is None:
                    return False

                # Check if the Alt key is pressed and re-select slices if so
                if not (0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]):
                    self.pixel_value = None
                    self.coord = None
                    self.info_label.setText(f"{orientation.capitalize()} - Coordinates: outside image area")
                    self.remove_mouse_highlights()
                elif QApplication.keyboardModifiers() == Qt.AltModifier:
                    self.update_slices_based_on_pixel(orientation, x, y)
                elif 0 <= x < slice_data.shape[1] and 0 <= y < slice_data.shape[0]:
                    # Highlight selected pixel in 3-d viewer (ignore if alt is pressed)
                    self.pixel_value = slice_data[y, x]
                    if orientation == 'axial':
                        self.coord = (y, x, self.axial_index)
                    elif orientation == 'coronal':
                        self.coord = (y, self.coronal_index, x)
                    elif orientation == 'sagittal':
                        self.coord = (self.sagittal_index, y, x)
                    self.highlight_voxel(self.coord, orientation)
                    self.info_label.setText(f"{orientation.capitalize()} - Coordinates: {self.coord} Value: {self.pixel_value:.2f}")
                return True

    def zoom_view(self, view, factor, cursor_pos):
            """
            Scales the view around the cursor position.

            Parameters:
                view (QGraphicsView): The view to zoom.
                factor (float): The scaling factor (>1.0 for zoom in, <1.0 for zoom out).
                cursor_pos (QPoint): The cursor position relative to the view widget.
            """
            # Map the viewport position to the scene's coordinate system
            scene_cursor_pos = view.mapToScene(cursor_pos)

            # Adjust the scene's scale and center based on the cursor's position
            old_transform = view.transform()
            new_transform = QTransform()
            new_transform.scale(factor, factor)
            new_transform = old_transform * new_transform

            # Set the new transformation to the view
            view.setTransform(new_transform)

    def update_slices_based_on_pixel(self, orientation, x, y):
            if self.data is None:
                return
            # Update the slice indices based on the orientation and coordinates
            if orientation == 'axial':
                if 0 <= x < self.data.shape[1] and 0 <= y < self.data.shape[0]:
                    self.coronal_index = x
                    self.sagittal_index = y
            elif orientation == 'coronal':
                if 0 <= x < self.data.shape[2] and 0 <= y < self.data.shape[0]:
                    self.axial_index = x
                    self.sagittal_index = y
            elif orientation == 'sagittal':
                if 0 <= x < self.data.shape[2] and 0 <= y < self.data.shape[1]:
                    self.axial_index = x
                    self.coronal_index = y

            # Update the planes and the slice views if indices are updated
            self.update_planes()
            self.display_all_slices()

    def handle_scroll(self, orientation, delta, scroll_speed):
            if self.data is None:
                return
            orientation_to_index = {'axial': 2, 'coronal': 1, 'sagittal': 0}
            max_index = self.data.shape[orientation_to_index[orientation]]
            index_attr = f"{orientation}_index"
            current_index = getattr(self, index_attr)
            if delta > 0:
                new_index = min(current_index + scroll_speed, max_index - 1)
            else:
                new_index = max(current_index - scroll_speed, 0)
            setattr(self, index_attr, new_index)

            # Only update planes if any are visible
            if any(self.plane_visibility.values()):
                self.update_planes()

            getattr(self, f"display_{orientation}_slice")()
