"""Core MRI segmentation viewer for the sseg package.

This module is adapted from the original standalone ``sSegEnv.py`` script and
preserves the existing viewer behavior while allowing package-based launch and
future refactoring.
"""

#Function to enable 3-d segmentation of MRI images
#Alex Ling (alling@bwh.harvard.edu; alexander.l.ling@gmail.com)
#12/3/2024
#E. Antonio Chiocca Group
#Brigham and Women's Hospital, Boston, MA, USA
#Generated with assistance of ChatGPT4 and ChatGPT o1-preview

import os
import sys
import argparse
import nibabel as nib
import nrrd
import numpy as np
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
import vtk
import re
import itertools
from vtk.util import numpy_support
from pyvista import Plane
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QGraphicsView, QGraphicsScene, QMenu, QWidgetAction,
                             QMenuBar, QAction, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QDialog, QPushButton, QGraphicsPixmapItem,
                             QComboBox, QToolBar, QMessageBox, QProgressDialog, QInputDialog, QLineEdit, QSpinBox, QGraphicsEllipseItem,
                             QFileDialog, QDockWidget, QTreeWidget, QTreeWidgetItem, QActionGroup)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent, QCursor, QPen, QColor, QPainter, QIntValidator, QDoubleValidator, QTransform
from scipy.ndimage import rotate, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, Normalize
from PyQt5.QtCore import Qt, QPoint, QEvent, QSettings, QTimer
from scipy.ndimage import binary_fill_holes
from queue import Queue
from functools import partial
from collections import deque

from .dialogs import ParameterDialog, RotationDialog
from .hotkeys import GlobalHotkeyFilter
from .icons import make_hand_icon
from .io_utils import (
    default_save_directory as compute_default_save_directory,
    determine_nrrd_space,
    get_rgb as color_name_to_rgb,
    load_and_reorient_volume,
    load_segmentation_data,
    reorient_affine_space_to_ras,
    reorient_to_original as reorient_volume_to_original,
    save_segmentation_nifti,
    save_segmentation_nrrd,
)
from .menus import PersistentMenu
from .rendering import RenderingMixin
from .segmentation_ops import SegmentationOpsMixin
from .slice_views import SliceViewMixin


class MRIViewer(RenderingMixin, SegmentationOpsMixin, SliceViewMixin, QMainWindow):
    def __init__(self, volume_paths, segmentation_paths, segmentation_suffixes, suffix_to_replace, ROI_mask_path, parent=None):
        super(MRIViewer, self).__init__(parent)
        
        #Initializing resize behavior
        self.resizeEvent = self.onResize

        # Initialize storage for currently selected coordinates and corresponding pixel value
        self.highlight_actors = {}
        self.coord = None
        self.pixel_value = None
        self.current_view = None

        # Initializing menu tool bar
        self.menu_bar = self.menuBar()
        # Menus with checkable items stay open while items are toggled (click elsewhere or press Esc to close)
        self.tools_menu = PersistentMenu("Tools", self, tear_off=False)
        self.menu_bar.addMenu(self.tools_menu)
        #self.rotation_action = QAction("Rotate Slices", self) #Currently omitting Rotate Slices option since I haven't had time to ensure rotations are propagated to all volumes/segmentation volumes and are reversible upon saving.
        #self.rotation_action.triggered.connect(self.open_rotation_dialog)
        #self.tools_menu.addAction(self.rotation_action)
        self.parameter_action = QAction("Adjust 3D Opacity", self)
        self.parameter_action.triggered.connect(self.open_parameter_dialog)
        self.tools_menu.addAction(self.parameter_action)

        # Performance options (remembered between sessions). The 3D hover marker is a single cached object and its renders are throttled,
        # so it is on by default; "Show 3D view" off skips all 3D work.

        settings = QSettings("sseg", "sseg")
        self.show_3d_hover_marker = settings.value("show_3d_hover_marker", True, type=bool)
        self.show_3d_view = settings.value("show_3d_view", True, type=bool)
        self.hover_marker_action = QAction("3D hover marker", self, checkable=True, checked=self.show_3d_hover_marker)
        self.hover_marker_action.toggled.connect(self.set_3d_hover_marker)
        self.tools_menu.addAction(self.hover_marker_action)

        # Display menu: choose which view windows are shown (and therefore rendered and interactive).
        self.view_enabled = {'3d': self.show_3d_view, 'axial': True, 'coronal': True, 'sagittal': True}
        self.display_menu = PersistentMenu("Display", self, tear_off=False)
        self.menu_bar.addMenu(self.display_menu)
        self.view_actions = {}
        for key, label in (('3d', "3D view"), ('axial', "Axial view"), ('coronal', "Coronal view"), ('sagittal', "Sagittal view")):
            action = QAction(label, self, checkable=True, checked=self.view_enabled[key])
            action.toggled.connect(lambda checked, k=key: self.set_view_enabled(k, checked))
            self.display_menu.addAction(action)
            self.view_actions[key] = action
        self.display_menu.addSeparator()
        # "Fast" renders the 3D view from a 2x downsampled copy of the volume (8x fewer voxels); 2D views are unaffected.
        self.quality_3d = settings.value("quality_3d", "full", type=str)
        if self.quality_3d not in ("full", "fast"):
            self.quality_3d = "full"
        quality_menu = PersistentMenu("3D quality", self, tear_off=False)
        self.tools_menu.addMenu(quality_menu)
        self.quality_action_group = QActionGroup(self)
        self.quality_action_group.setExclusive(True)
        for mode, label in (("full", "Full resolution"), ("fast", "Fast (half resolution)")):
            action = QAction(label, self, checkable=True, checked=(self.quality_3d == mode))
            action.triggered.connect(lambda _checked, m=mode: self.set_3d_quality(m))
            self.quality_action_group.addAction(action)
            quality_menu.addAction(action)
        self.plane_visibility = {
            'axial': False,
            'coronal': False,
            'sagittal': False
        }
        plane_menu = PersistentMenu("Toggle 3d Planes", self)
        self.add_plane_toggle_action(plane_menu, 'Axial', 'axial')
        self.add_plane_toggle_action(plane_menu, 'Coronal', 'coronal')
        self.add_plane_toggle_action(plane_menu, 'Sagittal', 'sagittal')
        self.tools_menu.addMenu(plane_menu)
        
        # First-level menu so segmentation levels can be shown/hidden with a single click on the menu bar
        self.segment_visibility_menu = PersistentMenu("Segmentation Levels", self, tear_off=False)
        self.menu_bar.addMenu(self.segment_visibility_menu)

        self.plane_meshes = {}

        # Track if the 3d viewer camera has been initialized yet
        self.new_camera = True

        # Initialize parameters for 3-d opacity calculations
        self.default_steepness = 20
        self.default_exponent = 3
        self.default_opacity_multiplier = 150
        self.steepness = self.default_steepness
        self.exponent = self.default_exponent
        self.opacity_multiplier = self.default_opacity_multiplier
        # Auto 3D opacity is currently deprecated because the recent automatic
        # transfer-function heuristic could make brains fully transparent.
        # Keep the flag for backwards compatibility, but default to manual.
        self.auto_3d_opacity = False

        self.view_axial = QGraphicsView()
        self.view_coronal = QGraphicsView()
        self.view_sagittal = QGraphicsView()
        for _view in (self.view_axial, self.view_coronal, self.view_sagittal):
            _view.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
            _view.setOptimizationFlags(QGraphicsView.DontSavePainterState | QGraphicsView.DontAdjustForAntialiasing)

        self.views = {
            'axial': self.view_axial,
            'coronal': self.view_coronal,
            'sagittal': self.view_sagittal
        }

        # Add UI components for volume and segmentation management
        self.widget = QWidget()
        self.layout = QGridLayout(self.widget)
        self.segmentation_tools = [
            "pan", "3d_contour", "3d_patch_contour", "3d_contour_global", "2d_brush",
            "grow_borders", "remove_volume", "keep_volume_only",
            "change_volume_level", "fill_holes"
        ]
        self.current_segmentation_tool = None
        self.segmentation_volumes = {}
        self.segmentation_names = {}
        self.current_segmentation = None
        self.current_segmentation_key = None
        self.unique_levels = np.array([0])
        self.invisible_levels = set()
        self.backup_invisible_levels = set()
        self.overlap_level = 0
        self.segmentation_level = 1
        self.overwrite_levels = set()
        self.new_segmentation_overlap = "add_around"
        self.is_drawing_2d_brush = False
        self.brush_cache_2d = None
        self.current_seg_slice_cache_2d = None
        self.brush_slice_cache = None
        self.brush_position = "Centered"
        self.color_map = [
            'red', 'green', 'blue', 'yellow', 'cyan',
            'magenta', 'orange', 'purple', 'brown', 'pink'
        ]
        self.level_color_map = {}
        self.actor_map = {}
        self.parameter_area = QWidget()
        self.layout.addWidget(self.parameter_area, 4, 0, 1, 2)
        self.init_volume_management_ui()
        self.init_segmentation_ui()

        #Initialize parameters for 3d plotting window
        self.plotter_3d = BackgroundPlotter(show=False)

        # Add ROI Loaded Label to the Layout (Row 0, Column 0, Span 1 Row and 2 Columns)
        self.roi_label = QLabel("ROI Loaded: None")
        self.layout.addWidget(self.roi_label, 0, 0, 1, 2)
        
        self.relayout_views()

        self.info_label = QLabel("Hover over image to see pixel info")
        self.layout.addWidget(self.info_label, 3, 0, 1, 2)

        self.setCentralWidget(self.widget)
        self.rotation_angles = {'x': 0, 'y': 0, 'z': 0}

        # Initialize tracking for image panning
        self.panning = False
        self.last_pan_point = QPoint()        

        # Set up the colored borders for 2D viewers
        self.setup_2d_viewer_borders()

        # Enable mouse tracking in views
        for view in self.views.values():
            view.setMouseTracking(True)
            view.viewport().setMouseTracking(True)
            view.viewport().installEventFilter(self)
        self.installEventFilter(self)
        self.global_hotkey_filter = GlobalHotkeyFilter(self)
        QApplication.instance().installEventFilter(self.global_hotkey_filter)

        # Loading volume data
        self.volume_paths = volume_paths
        self.segmentation_paths = segmentation_paths
        self.segmentation_suffixes = segmentation_suffixes
        self.suffix_to_replace = suffix_to_replace
        self.ROI_mask_path = ROI_mask_path

        self.mri_volumes = {}
        self.selected_volume = None
        self.voxel_dims = None
        self.data = None
        self.global_min = None
        self.global_max = None

        self.axial_index = None
        self.coronal_index = None
        self.sagittal_index = None

        self.master_shape = None
        self.master_affine = None
        self.master_orientation = None
        self.master_voxel_dims = None
        self.master_canonical_affine = None
        self.master_canonical_orientation = None
        self.master_canonical_voxel_dims = None
        self.master_canonical_shape = None

        # Undo/redo and dirty-tracking state must exist before any volume or segmentation is loaded.
        self.undo_stack = []
        self.redo_stack = []
        self.segmentation_dirty = {}

        self.initial_load = True
        if self.volume_paths:
            for path in self.volume_paths:
                self.load_volume(path)
            self.switch_volume(list(self.mri_volumes.keys())[0])
        self.initial_load = False

        if self.segmentation_paths:
            for path in self.segmentation_paths:
                self.load_segmentation(path)

        # Initialize ROI mask
        if self.ROI_mask_path:
            self.load_roi(self.ROI_mask_path[0])
        else:
            self.roi_mask = None

        # Ensure the viewer supports an intentionally empty session state.
        if self.data is None:
            self.show_empty_state()

        self.update_segment_visibility_menu()
        # Handle undo and redo operations (state is initialized before loading, see above)
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        self.addAction(undo_action)
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo)
        self.addAction(redo_action)

        # Storage for segmentation volumes
        if self.volume_paths:
            self.file_name = os.path.basename(self.volume_paths[0])
        elif self.segmentation_paths:
            self.file_name = os.path.basename(self.segmentation_paths[0])
        else:
            self.file_name = ""

        if self.segmentation_suffixes and self.suffix_to_replace:
            for suffix in self.segmentation_suffixes:
                if not self.suffix_to_replace[0] in self.file_name:
                    print(f"Waring: Provided --suffix_to_replace ({self.suffix_to_replace[0]}) not found in first loaded filename ({self.file_name}).")
                seg_name = self.file_name.replace(self.suffix_to_replace[0], suffix)
                self.create_new_segmentation(seg_name)
        elif self.segmentation_suffixes:
            for suffix in self.segmentation_suffixes:
                seg_name = self.file_name + suffix
                self.create_new_segmentation(seg_name)

        # Initial display of all slices and planes
        if self.segmentation_volumes.keys():
            self.handle_segmentation_selection(list(self.segmentation_volumes.keys())[0])
            self.showMaximized()
            self.display_all_slices()
        else:
            self.handle_segmentation_selection(None)
            self.showMaximized()
            self.display_all_slices()

    def _clear_combo_items(self, combo, keep_items=None):
        keep_items = list(keep_items or [])
        combo.blockSignals(True)
        combo.clear()
        for item in keep_items:
            combo.addItem(item)
        combo.blockSignals(False)

    def reset_2d_view_state(self):
        for view in [self.view_axial, self.view_coronal, self.view_sagittal]:
            try:
                view.resetTransform()
            except Exception:
                pass
            for attr in ('pixmap_item', 'aspect_ratio', 'current_scale'):
                if hasattr(view, attr):
                    try:
                        delattr(view, attr)
                    except Exception:
                        pass
            scene = view.scene()
            if scene is not None:
                try:
                    scene.clear()
                except Exception:
                    pass

    def show_empty_state(self):
        self.reset_2d_view_state()
        self.coord = None
        self.pixel_value = None
        self.info_label.setText("No exam loaded")
        self.roi_label.setText("ROI Loaded: None")
        self.remove_mouse_highlights()
        self.plotter_3d.clear()
        self.plane_meshes = {}
        self.actor_map = {}
        self.display_placeholder_views("No exam loaded")

    def clear_loaded_exam(self):
        # Clear currently loaded exam-specific data so the viewer can remain open
        # with no active MRI volumes. This is needed for batch exam navigation.
        self.remove_mouse_highlights()
        self.current_view = None
        self.mri_volumes = {}
        self.selected_volume = None
        self.volume_paths = []
        self.data = None
        self.affine = None
        self.voxel_dims = None
        self.original_data = None
        self.original_affine = None
        self.original_voxel_dims = None
        self.global_min = None
        self.global_max = None
        self.axial_index = None
        self.coronal_index = None
        self.sagittal_index = None

        self.master_shape = None
        self.master_affine = None
        self.master_orientation = None
        self.master_voxel_dims = None
        self.master_canonical_affine = None
        self.master_canonical_orientation = None
        self.master_canonical_voxel_dims = None
        self.master_canonical_shape = None

        self.segmentation_volumes = {}
        self.segmentation_names = {}
        self.current_segmentation = None
        self.current_segmentation_key = None
        self.unique_levels = np.array([0])
        self.invisible_levels = set()
        self.backup_invisible_levels = set()
        self.overlap_level = 0
        self.overwrite_levels = set()
        self.segmentation_paths = []
        self.roi_mask = None
        self.undo_stack = []
        self.redo_stack = []
        self.segmentation_dirty = {}
        self.batch_current_exam = None

        self._clear_combo_items(self.volume_combo)
        self._clear_combo_items(self.segmentation_combo, keep_items=["None", "Create new segmentation"])
        self.segmentation_combo.setCurrentText("None")
        self.update_level_name_display()
        self.update_segment_visibility_menu()
        self.update_overwrite_levels_menu()
        self.update_overlap_level_menu()
        self.show_empty_state()

    def has_unsaved_segmentations(self):
        return any(bool(v) for v in self.segmentation_dirty.values())

    def mark_segmentation_dirty(self, key=None, dirty=True):
        key = self.current_segmentation_key if key is None else key
        if key:
            self.segmentation_dirty[key] = bool(dirty)

    def prompt_save_discard_or_cancel(self, title="Unsaved Changes", message=None):
        if not self.has_unsaved_segmentations():
            return "discard"
        if message is None:
            message = "There are unsaved segmentation changes in the current environment. Do you want to save them before continuing?"
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        save_btn = msg_box.addButton("Save", QMessageBox.AcceptRole)
        discard_btn = msg_box.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_btn = msg_box.addButton(QMessageBox.Cancel)
        msg_box.setDefaultButton(save_btn)
        msg_box.exec_()
        clicked = msg_box.clickedButton()
        if clicked == save_btn:
            return "save"
        if clicked == discard_btn:
            return "discard"
        return "cancel"

    def save_dirty_segmentations_interactive(self):
        dirty_keys = [key for key, dirty in self.segmentation_dirty.items() if dirty and key in self.segmentation_volumes]
        if not dirty_keys:
            return True
        original_key = self.current_segmentation_key
        for key in dirty_keys:
            if key not in self.segmentation_volumes:
                continue
            self.switch_segmentation(key)
            self.save_current_segmentation()
            if self.segmentation_dirty.get(key, False):
                if original_key in self.segmentation_volumes:
                    self.switch_segmentation(original_key)
                return False
        if original_key in self.segmentation_volumes:
            self.switch_segmentation(original_key)
        return True

    def confirm_switch_exam_with_unsaved_changes(self, next_exam_label=None):
        if not self.has_unsaved_segmentations():
            return True
        message = "There are unsaved segmentation changes in the current exam."
        if next_exam_label:
            message += f" Save them before loading '{next_exam_label}'?"
        else:
            message += " Save them before switching exams?"
        choice = self.prompt_save_discard_or_cancel(title="Unsaved Changes", message=message)
        if choice == "cancel":
            return False
        if choice == "discard":
            return True
        return self.save_dirty_segmentations_interactive()

    def confirm_discard_unsaved_segmentations(self, title="Unsaved Changes", message=None):
        if not self.has_unsaved_segmentations():
            return True
        if message is None:
            message = "There are unsaved segmentation changes in the current environment. Clear the environment and discard those changes?"
        reply = QMessageBox.question(self, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def show_batch_dock(self):
        dock = getattr(self, 'batch_dock', None)
        if dock is None:
            return
        dock.show()
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()

    def initialize_batch_navigation(self):
        if not getattr(self, 'batch_discovery_result', None):
            return
        if getattr(self, 'batch_dock', None) is None:
            self.batch_dock = QDockWidget("Batch Exams", self)
            self.batch_dock.setObjectName("Batch Exams")
            self.batch_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

            container = QWidget()
            layout = QVBoxLayout(container)
            self.batch_status_label = QLabel("Select an exam and click Load.")
            self.batch_status_label.setWordWrap(True)
            layout.addWidget(self.batch_status_label)

            self.batch_tree = QTreeWidget()
            self.batch_tree.setHeaderHidden(True)
            self.batch_tree.itemSelectionChanged.connect(self.update_batch_selection_label)
            self.batch_tree.itemDoubleClicked.connect(lambda *_args: self.load_selected_batch_exam())
            layout.addWidget(self.batch_tree)

            button_row = QHBoxLayout()
            self.batch_load_button = QPushButton("Load Selected Exam")
            self.batch_load_button.clicked.connect(self.load_selected_batch_exam)
            button_row.addWidget(self.batch_load_button)
            refresh_btn = QPushButton("Refresh List")
            refresh_btn.clicked.connect(self.populate_batch_exam_tree)
            button_row.addWidget(refresh_btn)
            layout.addLayout(button_row)

            self.batch_dock.setWidget(container)
            self.addDockWidget(Qt.LeftDockWidgetArea, self.batch_dock)

            # Toolbar button and Display-menu entry to bring the exam list back if it was closed
            self.batch_show_action = QAction("Exam List", self)
            self.batch_show_action.setToolTip("Show the multi-patient exam selection window")
            self.batch_show_action.triggered.connect(self.show_batch_dock)
            self.toolbar.insertAction(self._pan_separator, self.batch_show_action)  # near the front so it never lands in the overflow menu
            self.display_menu.addAction(self.batch_dock.toggleViewAction())
            self.batch_dock.toggleViewAction().setText("Exam list")
        self.populate_batch_exam_tree()

    def set_3d_hover_marker(self, enabled):
        self.show_3d_hover_marker = bool(enabled)
        QSettings("sseg", "sseg").setValue("show_3d_hover_marker", self.show_3d_hover_marker)
        if not enabled:
            self.hide_hover_markers()

    def set_3d_quality(self, mode):
        if mode == self.quality_3d:
            return
        self.quality_3d = mode
        QSettings("sseg", "sseg").setValue("quality_3d", mode)
        if self.data is not None and self.show_3d_view:
            self.update_visualization(reset_volume=True)

    def relayout_views(self):
        """Place the enabled views in the grid so they fill the available space (4 -> 2x2, 3 -> 2 + 1 wide, 2 -> side by side)."""
        widgets = [('3d', self.plotter_3d.interactor), ('axial', self.view_axial),
                   ('coronal', self.view_coronal), ('sagittal', self.view_sagittal)]
        for _key, widget in widgets:
            self.layout.removeWidget(widget)
        shown = [widget for key, widget in widgets if self.view_enabled[key]]
        for key, widget in widgets:
            if not self.view_enabled[key]:
                widget.setVisible(False)
        placements = {
            1: [(1, 0, 2, 2)],
            2: [(1, 0, 2, 1), (1, 1, 2, 1)],
            3: [(1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 2)],
            4: [(1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1), (2, 1, 1, 1)],
        }.get(len(shown), [])
        for widget, (row, col, row_span, col_span) in zip(shown, placements):
            self.layout.addWidget(widget, row, col, row_span, col_span)
            widget.setVisible(True)
        for row in (1, 2):
            self.layout.setRowStretch(row, 1)
        for col in (0, 1):
            self.layout.setColumnStretch(col, 1)
        QTimer.singleShot(0, self.fit_views_to_contents)

    def set_view_enabled(self, key, enabled):
        if not enabled and not any(v for k, v in self.view_enabled.items() if k != key):
            # Keep at least one view on screen
            self.view_actions[key].blockSignals(True)
            self.view_actions[key].setChecked(True)
            self.view_actions[key].blockSignals(False)
            return
        self.view_enabled[key] = bool(enabled)
        if key == '3d':
            self.set_3d_view_visible(enabled)
            return
        self.relayout_views()
        if enabled and self.data is not None:
            getattr(self, f"display_{key}_slice")()  # views skip drawing while disabled, so refresh on re-enable

    def set_3d_view_visible(self, visible):
        self.show_3d_view = bool(visible)
        self.view_enabled['3d'] = self.show_3d_view
        QSettings("sseg", "sseg").setValue("show_3d_view", self.show_3d_view)
        self.relayout_views()
        if self.show_3d_view and self.data is not None:
            self.update_visualization(reset_volume=True)

    def get_batch_visible_exams(self):
        result = getattr(self, 'batch_discovery_result', None)
        selection = getattr(self, 'batch_launch_selection', None)
        if result is None or selection is None:
            return []
        suffix = (selection.selected_suffix or '').strip()
        if not suffix:
            return []
        selected_series = list(selection.selected_series_types or [])
        require_all = bool(selection.require_all_selected_series)
        exams = []
        for exam in result.exams:
            if suffix not in exam.available_suffixes:
                continue
            if require_all and selected_series:
                available = set(exam.available_series_types_by_suffix.get(suffix, []))
                if not all(series in available for series in selected_series):
                    continue
            exams.append(exam)
        exams.sort(key=lambda exam: (exam.patient_id.lower(), exam.exam_id.lower(), exam.exam_dir.lower()))
        return exams

    def populate_batch_exam_tree(self):
        if getattr(self, 'batch_tree', None) is None:
            return
        self.batch_tree.clear()
        exams = self.get_batch_visible_exams()
        root_dir = ''
        if getattr(self, 'batch_discovery_result', None) is not None:
            root_dir = getattr(self.batch_discovery_result.settings, 'root_dir', '') or ''
        by_patient = {}
        for exam in exams:
            patient_item = by_patient.get(exam.patient_id)
            if patient_item is None:
                patient_item = QTreeWidgetItem([exam.patient_id])
                patient_item.setData(0, Qt.UserRole, None)
                by_patient[exam.patient_id] = patient_item
                self.batch_tree.addTopLevelItem(patient_item)
            rel_path = exam.exam_dir
            if root_dir:
                try:
                    rel_path = os.path.relpath(exam.exam_dir, root_dir)
                except Exception:
                    rel_path = exam.exam_dir
            suffix = (self.batch_launch_selection.selected_suffix or '').strip() if getattr(self, 'batch_launch_selection', None) else ''
            available_series = exam.available_series_types_by_suffix.get(suffix, []) if suffix else []
            child = QTreeWidgetItem([exam.exam_id])
            child.setToolTip(0, f"{rel_path}\n{exam.exam_dir}\nSeries: {', '.join(available_series) if available_series else 'none'}")
            child.setData(0, Qt.UserRole, exam.exam_dir)
            patient_item.addChild(child)
            patient_item.setExpanded(True)
        self.batch_tree.expandAll()
        self.update_batch_selection_label()
        if getattr(self, 'batch_load_button', None) is not None:
            self.batch_load_button.setEnabled(bool(exams))

    def get_selected_batch_exam(self):
        if getattr(self, 'batch_tree', None) is None:
            return None
        item = self.batch_tree.currentItem()
        if item is None:
            return None
        exam_dir = item.data(0, Qt.UserRole)
        if not exam_dir:
            return None
        result = getattr(self, 'batch_discovery_result', None)
        if result is None:
            return None
        for exam in result.exams:
            if exam.exam_dir == exam_dir:
                return exam
        return None

    def update_batch_selection_label(self):
        if getattr(self, 'batch_status_label', None) is None:
            return
        exam = self.get_selected_batch_exam()
        if exam is None:
            self.batch_status_label.setText("Select an exam and click Load.")
            return
        selection = getattr(self, 'batch_launch_selection', None)
        suffix = (selection.selected_suffix or '').strip() if selection else ''
        selected_series = list(selection.selected_series_types or []) if selection else []
        available = exam.available_series_types_by_suffix.get(suffix, []) if suffix else []
        planned = [series for series in selected_series if series in available] if selected_series else list(available)
        self.batch_status_label.setText(
            f"Selected exam: {exam.exam_id}\n"
            f"Suffix: {suffix or 'none'}\n"
            f"Series to load: {', '.join(planned) if planned else 'none'}"
        )

    def get_batch_exam_load_plan(self, exam):
        selection = getattr(self, 'batch_launch_selection', None)
        suffix = (selection.selected_suffix or '').strip() if selection else ''
        selected_series = list(selection.selected_series_types or []) if selection else []
        preferred_group = exam.preferred_group_by_suffix.get(suffix)
        series_files = []
        for sf in exam.series_files:
            if sf.shared_suffix != suffix:
                continue
            if preferred_group and sf.compatible_group_key != preferred_group:
                continue
            if selected_series and sf.series_type not in selected_series:
                continue
            series_files.append(sf)
        if selected_series:
            order = {name: idx for idx, name in enumerate(selected_series)}
            series_files.sort(key=lambda sf: (order.get(sf.series_type, len(order)), sf.filename.lower()))
        else:
            series_files.sort(key=lambda sf: (sf.series_type.lower(), sf.filename.lower()))
        volume_paths = [sf.path for sf in series_files]

        replacement_suffix = (selection.replacement_segmentation_suffix or '').strip() if selection else ''
        segmentation_paths = []
        # Always load the preferred existing segmentation when one is available.
        # If a replacement suffix is requested, the loaded segmentation will be
        # copied/renamed into the replacement output name rather than skipped.
        if exam.preferred_segmentation_path:
            segmentation_paths = [exam.preferred_segmentation_path]

        return {
            'volume_paths': volume_paths,
            'roi_mask_path': exam.brainmask_path,
            'segmentation_paths': segmentation_paths,
            'replacement_segmentation_suffix': replacement_suffix,
        }

    def create_batch_replacement_segmentation(self, exam, replacement_suffix, source_segmentation_path=None):
        replacement_suffix = (replacement_suffix or '').strip()
        if not replacement_suffix:
            return
        if replacement_suffix.startswith('_'):
            seg_name = f"{exam.exam_id}{replacement_suffix}"
        else:
            seg_name = f"{exam.exam_id}_{replacement_suffix}"
        if seg_name in self.segmentation_volumes:
            self.switch_segmentation(seg_name)
            return

        source_key = os.path.basename(source_segmentation_path) if source_segmentation_path else None
        if source_key and source_key in self.segmentation_volumes:
            # In batch mode, a replacement suffix should preserve the loaded segmentation
            # contents and simply prepare a new output-named copy for editing/saving.
            self.segmentation_volumes[seg_name] = np.array(self.segmentation_volumes[source_key], copy=True)
            self.segmentation_names[seg_name] = dict(self.segmentation_names.get(source_key, {}))
            self.segmentation_dirty[seg_name] = True
            self.segmentation_combo.addItem(seg_name)
            self.switch_segmentation(seg_name)

            # Remove the source segmentation from the viewer so the replacement truly
            # acts like a rename rather than leaving a duplicate loaded entry around.
            if source_key != seg_name:
                self.segmentation_volumes.pop(source_key, None)
                self.segmentation_names.pop(source_key, None)
                self.segmentation_dirty.pop(source_key, None)
                idx = self.segmentation_combo.findText(source_key)
                if idx >= 0:
                    self.segmentation_combo.removeItem(idx)
            self.segmentation_paths = []
            return

        self.create_new_segmentation(seg_name)
        self.mark_segmentation_dirty(seg_name, True)

    def load_selected_batch_exam(self):
        exam = self.get_selected_batch_exam()
        if exam is None:
            QMessageBox.information(self, "Batch Exams", "Please select an exam to load.")
            return
        if not self.confirm_switch_exam_with_unsaved_changes(next_exam_label=exam.exam_id):
            return
        plan = self.get_batch_exam_load_plan(exam)
        volume_paths = plan.get('volume_paths', [])
        if not volume_paths:
            QMessageBox.warning(self, "Batch Exams", f"No compatible series volumes were found to load for '{exam.exam_id}'.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.clear_loaded_exam()
            self.batch_current_exam = exam
            self.file_name = os.path.basename(volume_paths[0]) if volume_paths else exam.exam_id

            # Preserve the first loaded batch volume as the active viewing volume.
            # load_volume() normally switches to each newly loaded volume when
            # initial_load is False, which would otherwise leave the last series
            # selected after a batch exam load.
            previous_initial_load = self.initial_load
            self.initial_load = True
            try:
                for volume_path in volume_paths:
                    self.load_volume(volume_path)
            finally:
                self.initial_load = previous_initial_load
            if volume_paths:
                first_volume_name = os.path.basename(volume_paths[0])
                if first_volume_name in self.mri_volumes:
                    self.switch_volume(first_volume_name)

            roi_mask_path = plan.get('roi_mask_path')
            if roi_mask_path:
                self.load_roi(roi_mask_path)
            segmentation_paths = plan.get('segmentation_paths', [])
            if segmentation_paths:
                self.load_segmentation(segmentation_paths)
            replacement_suffix = plan.get('replacement_segmentation_suffix', '')
            if replacement_suffix:
                source_segmentation_path = segmentation_paths[0] if segmentation_paths else None
                self.create_batch_replacement_segmentation(exam, replacement_suffix, source_segmentation_path=source_segmentation_path)
            if self.data is not None:
                self.display_all_slices()
                self.update_visualization(reset_volume=True)
                self.fit_views_to_contents()
        finally:
            QApplication.restoreOverrideCursor()

        if getattr(self, 'batch_status_label', None) is not None:
            loaded_series = [os.path.basename(path) for path in volume_paths]
            self.batch_status_label.setText(
                f"Loaded exam: {exam.exam_id}\n"
                f"Volumes: {', '.join(loaded_series)}\n"
                f"ROI: {os.path.basename(plan.get('roi_mask_path')) if plan.get('roi_mask_path') else 'none'}"
            )

    def clear_environment(self):
        if not self.confirm_discard_unsaved_segmentations(title="Clear Environment"):
            return
        self.clear_loaded_exam()

    def onResize(self, event):
        super().resizeEvent(event)
        self.fit_views_to_contents()

    def fit_views_to_contents(self):
        for view in [self.view_axial, self.view_coronal, self.view_sagittal]:
            scene = view.scene()
            if scene is None:
                continue
            bounds = scene.itemsBoundingRect()
            if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
                bounds = scene.sceneRect()
            if bounds.isNull() or bounds.width() <= 0 or bounds.height() <= 0:
                continue
            view.setSceneRect(bounds)
            view.fitInView(bounds, Qt.KeepAspectRatio)

    def init_volume_management_ui(self):
        # Create the "Manage Volumes" menu in the menu bar
        manage_volumes_menu = QMenu("Manage Volumes", self)
        
        # Create actions for the menu
        load_volume_action = QAction("Load Volume", self)
        load_volume_action.triggered.connect(self.prompt_load_volume)
        manage_volumes_menu.addAction(load_volume_action)
        
        unload_volume_action = QAction("Unload Volume", self)
        unload_volume_action.triggered.connect(self.delete_volume)
        manage_volumes_menu.addAction(unload_volume_action)
        
        rename_volume_action = QAction("Rename Volume", self)
        rename_volume_action.triggered.connect(self.rename_volume)
        manage_volumes_menu.addAction(rename_volume_action)
        
        # Add the "Manage Volumes" menu to the menu bar
        self.menu_bar.addMenu(manage_volumes_menu)

    # Code for loading and saving segmentations and volumes

    def prompt_load_volume(self, checked=False):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load MRI Volume(s)",
            self.default_save_directory(),
            "Images (*.nii *.nii.gz *.nrrd);;All files (*)"
        )
        for file_path in file_paths:
            if file_path:
                self.load_volume(file_path)

    def load_volume(self, file_path):
        data, affine, voxel_dims = self.load_and_reorient_to_RAS(file_path)
        if data is not None:
            volume_name = os.path.basename(file_path)
            self.mri_volumes[volume_name] = (data, affine, voxel_dims, data, affine, voxel_dims)
            if file_path not in self.volume_paths:
                self.volume_paths.append(file_path)
            if self.volume_combo.findText(volume_name) < 0:
                self.volume_combo.addItem(volume_name)
            if self.selected_volume is None or not self.initial_load:
                self.switch_volume(volume_name)

    def quicksave_to_nrrd(self):
        current_key = self.current_segmentation_key
        for seg_name in self.segmentation_volumes.keys():
            try:
                file_path = os.path.join(self.default_save_directory(), seg_name)
                file_path = re.sub(r'\.seg\.nrrd$', '', file_path) # Strip .seg.nrrd extension if it exists
                file_path = re.sub(r'\.nrrd$', '', file_path) # Strip .nrrd extension if it exists
                file_path = re.sub(r'\.nii$', '', file_path) # Strip .nii extension if it exists
                file_path = re.sub(r'\.nii.gz$', '', file_path) # Strip .nii.gz extension if it exists
                file_path = file_path + '.seg.nrrd'
                self.switch_segmentation(seg_name)
                self.save_as_nrrd(file_path)
                self.mark_segmentation_dirty(seg_name, False)
            except Exception as e:
                QMessageBox.critical(self, "Quicksave Error", f"Failed to quicksave segmentation '{seg_name}':\n{str(e)}")
        self.switch_segmentation(current_key)

    def quicksave_to_nifti(self):
        current_key = self.current_segmentation_key
        for seg_name in self.segmentation_volumes.keys():
            try:
                file_path = os.path.join(self.default_save_directory(), seg_name)
                file_path = re.sub(r'\.seg\.nrrd$', '', file_path) # Strip .seg.nrrd extension if it exists
                file_path = re.sub(r'\.nrrd$', '', file_path) # Strip .nrrd extension if it exists
                file_path = re.sub(r'\.nii$', '', file_path) # Strip .nii extension if it exists
                file_path = re.sub(r'\.nii.gz$', '', file_path) # Strip .nii.gz extension if it exists
                file_path = file_path + '.nii.gz'
                self.switch_segmentation(seg_name)
                self.save_as_nifti(file_path)
                self.mark_segmentation_dirty(seg_name, False)
            except Exception as e:
                QMessageBox.critical(self, "Quicksave Error", f"Failed to quicksave segmentation '{seg_name}':\n{str(e)}")
        self.switch_segmentation(current_key)

    def save_current_segmentation(self):
        if not self.current_segmentation_key or self.current_segmentation is None:
            QMessageBox.warning(self, "Save Segmentation", "No segmentation selected or loaded to save.")
            return

        format_choices = "NRRD File (*.seg.nrrd);;Nifti File (*.nii.gz)"
        default_filename = os.path.join(self.default_save_directory(), self.current_segmentation_key)

        # Sanitize default filename
        default_filename = re.sub(r'\.seg\.nrrd$|\.nrrd$|\.nii$|\.nii\.gz$', '', default_filename)
        
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Segmentation", default_filename, format_choices)

        if not file_path:
            return  # User cancelled the save dialog

        # Append file extension if missing
        if selected_filter == "Nifti File (*.nii.gz)":
            if not (file_path.endswith('.nii.gz') or file_path.endswith('.nii')):
                file_path += '.nii.gz'
            self.save_as_nifti(file_path)
        elif selected_filter == "NRRD File (*.seg.nrrd)":
            if not file_path.endswith('.seg.nrrd'):
                file_path += '.seg.nrrd'
            self.save_as_nrrd(file_path)
        else:
            QMessageBox.warning(self, "Save Segmentation", "Unrecognized file format selected.")

    def default_save_directory(self):
        return compute_default_save_directory(self.volume_paths, self.segmentation_paths)

    def save_as_nifti(self, file_path):
        try:
            save_segmentation_nifti(
                self.current_segmentation,
                file_path,
                master_canonical_affine=self.master_canonical_affine,
                master_affine=self.master_affine,
            )
            QMessageBox.information(self, "Save Successful", f"Segmentation saved successfully:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save segmentation as NIfTI:\n{str(e)}")

    def get_rgb(self, color_name):
        return color_name_to_rgb(color_name)

    def save_as_nrrd(self, file_path):
        if self.current_segmentation is None:
            QMessageBox.warning(self, "Error", "No segmentation available to save.")
            return

        try:
            save_segmentation_nrrd(
                self.current_segmentation,
                file_path,
                master_canonical_affine=self.master_canonical_affine,
                master_affine=self.master_affine,
                unique_levels=self.unique_levels,
                level_names=self.segmentation_names.get(self.current_segmentation_key, {}),
            )
            QMessageBox.information(self, "Save Successful", f"Segmentation saved successfully:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save segmentation: {str(e)}")

    # Code for volume management

    def switch_volume(self, name):
        if name in self.mri_volumes:
            if self.data is not None:
                old_data_shape = self.data.shape
            else:
                old_data_shape = None
            self.data, self.affine, self.voxel_dims, self.original_data, self.original_affine, self.original_voxel_dims = self.mri_volumes[name]
            self.global_min = self.original_data.min()
            self.global_max = self.original_data.max()
            self.selected_volume = name
            self.volume_combo.setCurrentText(self.selected_volume)
            reinitialize_views = (old_data_shape is None) or (self.data.shape != old_data_shape)
            if None in (self.axial_index, self.coronal_index, self.sagittal_index) or reinitialize_views:
                self.axial_index = self.data.shape[2] // 2
                self.coronal_index = self.data.shape[1] // 2
                self.sagittal_index = self.data.shape[0] // 2

            if reinitialize_views:
                self.reset_2d_view_state()

            self.update_visualization(reset_volume=True)
            self.display_all_slices()
            if reinitialize_views:
                self.fit_views_to_contents()
        elif name is None:
            self.selected_volume = None
            self.data = None
            self.affine = None
            self.voxel_dims = None
            self.original_data = None
            self.original_affine = None
            self.original_voxel_dims = None
            self.global_min = None
            self.global_max = None
            self.axial_index = None
            self.coronal_index = None
            self.sagittal_index = None
            self.show_empty_state()
        else:
            QMessageBox.information(self, "Volume Error", f"Attempted to switch to MRI volume '{name}' that does not exist.")

    def update_volume(self):
        if self.selected_volume in self.mri_volumes:
            self.mri_volumes[self.selected_volume] = (self.data, self.affine, self.voxel_dims, self.original_data, self.original_affine, self.original_voxel_dims)
        else:
            QMessageBox.information(self, "Volume Update Error", f"Attempted to update MRI volume '{self.selected_volume}' that does not exist.")

    def delete_volume(self):
        if not self.selected_volume:
            QMessageBox.information(self, "Unload Volume", "No MRI volume is currently loaded.")
            return

        reply = QMessageBox.question(self, 'Confirm Unload',
                                    f"Unload '{self.selected_volume}' from the viewer? This will not delete anything from disk.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            deleted_name = self.selected_volume
            del self.mri_volumes[deleted_name]
            index = self.volume_combo.findText(deleted_name)
            self.volume_combo.removeItem(index)
            if self.mri_volumes:
                new_index = max(0, index - 1)
                if new_index < self.volume_combo.count():
                    self.switch_volume(self.volume_combo.itemText(new_index))
                else:
                    self.switch_volume(next(iter(self.mri_volumes.keys())))
            else:
                self.clear_loaded_exam()

    def rename_volume(self):
        if self.selected_volume:
            new_name, ok = QInputDialog.getText(self, 'Rename Volume',
                                                'Enter new name for the MRI volume:',
                                                text=self.selected_volume)
            if ok and new_name:
                self.mri_volumes[new_name] = self.mri_volumes.pop(self.selected_volume)
                index = self.volume_combo.findText(self.selected_volume)
                self.volume_combo.setItemText(index, new_name)
                self.selected_volume = new_name
                QMessageBox.information(self, "Rename Volume", "Volume renamed successfully!")

    # Code for managing undo/redo actions

    # Undo/redo store only the region an edit touched: entries are (segmentation key, bbox, voxel values of that region).
    def _swap_region(self, stack_from, stack_to, empty_message, none_for_key_message, action_name):
        if not stack_from:
            QMessageBox.information(self, action_name, empty_message)
            return
        indices = [i for i, entry in enumerate(stack_from) if entry[0] == self.current_segmentation_key]
        if not indices:
            QMessageBox.information(self, action_name, none_for_key_message)
            return
        key, bbox, stored = stack_from.pop(indices[-1])
        region = tuple(slice(lo, hi) for lo, hi in bbox)
        seg = self.current_segmentation
        if seg is None or seg[region].shape != stored.shape:
            QMessageBox.warning(self, action_name, "This history entry no longer matches the loaded segmentation and was discarded.")
            return
        stack_to.append((key, bbox, np.copy(seg[region])))  # what the region holds now, for the opposite operation
        seg[region] = stored
        self.segmentation_volumes[self.current_segmentation_key] = seg

        # Update the visualization and display
        self.mark_segmentation_dirty(self.current_segmentation_key, True)
        self.refresh_unique_levels()
        self.update_level_menus_if_changed()
        self.update_visualization(reset_volume=False, dirty_bbox=bbox)
        self.display_all_slices()

    def undo(self):
        self._swap_region(self.undo_stack, self.redo_stack, "No actions to undo.",
                          "No actions to undo for the current segmentation.", "Undo")

    def redo(self):
        self._swap_region(self.redo_stack, self.undo_stack, "No actions to redo.",
                          "No actions to redo for the current segmentation.", "Redo")
        if self.current_segmentation_key:
            self.segmentation_combo.setCurrentText(self.current_segmentation_key)

    def push_to_undo_stack(self, bbox=None):
        # Store the pre-edit values of the region about to change (the whole volume if no bbox is given)
        if self.current_segmentation is not None:
            if bbox is None:
                bbox = tuple((0, n) for n in self.current_segmentation.shape)
            region = tuple(slice(lo, hi) for lo, hi in bbox)
            # A new edit invalidates anything that could have been redone for this segmentation
            self.redo_stack[:] = [e for e in self.redo_stack if e[0] != self.current_segmentation_key]
            # Check that undo stack isn't getting too big
            if len(self.undo_stack) >= 20:
                self.undo_stack.pop(0)  # Remove oldest undo stack state if there are 20 or more states in the stack
            # np.copy ensures we store a snapshot, not a view
            self.undo_stack.append((self.current_segmentation_key, tuple(bbox), np.copy(self.current_segmentation[region])))

    def refresh_unique_levels(self, changed_region=None):
        """Recompute self.unique_levels exactly. After an edit, only levels that were absent from the changed region
        need a whole-volume check, which avoids a full np.unique pass in the common case."""
        seg = self.current_segmentation
        if seg is None:
            self.unique_levels = np.array([0])
            return
        if changed_region is None:
            self.unique_levels = np.unique(seg)
            return
        in_region = np.unique(seg[changed_region])
        levels = set(int(l) for l in in_region)
        for level in self.unique_levels:
            level = int(level)
            if level not in levels and (seg == level).any():
                levels.add(level)
        self.unique_levels = np.array(sorted(levels), dtype=seg.dtype)

    def update_level_menus_if_changed(self):
        key = (tuple(int(l) for l in self.unique_levels), self.segmentation_level_spinbox.value(),
               frozenset(self.invisible_levels), frozenset(self.overwrite_levels))
        if key == getattr(self, '_level_menu_key', None):
            return
        self.update_overlap_level_menu()
        self.update_segment_visibility_menu()
        self.update_overwrite_levels_menu()
        self._level_menu_key = key

    def update_segmentation_level(self):
        # Update the segmentation level based on the spinbox value
        self.segmentation_level = self.segmentation_level_spinbox.value()

    def toggle_segment_level_visibility(self, level, visible):
        if visible:
            self.invisible_levels.discard(level)  # Remove from invisible levels if toggled on
        else:
            self.invisible_levels.add(level)  # Add to invisible levels if toggled off
        self.schedule_visibility_redraw()

    def schedule_visibility_redraw(self):
        """Redraw once, shortly after the last visibility change, so several quick toggles cost a single re-render."""
        timer = getattr(self, '_visibility_timer', None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(60)
            timer.timeout.connect(self._apply_visibility_redraw)
            self._visibility_timer = timer
        timer.start()

    def _apply_visibility_redraw(self):
        if self.data is None:
            return
        self.update_visualization(reset_volume=False)  # 3D overlay refresh is debounced separately
        self.display_all_slices()

    def set_all_levels_visible(self, visible):
        self.invisible_levels = set() if visible else set(int(l) for l in self.unique_levels if l != 0)
        for action in self.segment_visibility_menu.actions():
            if action.isCheckable():
                action.blockSignals(True)
                action.setChecked(visible)
                action.blockSignals(False)
        self.schedule_visibility_redraw()

    def update_segment_visibility_menu(self):
        self._level_menu_key = None  # menus rebuilt outside update_level_menus_if_changed
        self.segment_visibility_menu.clear()
        has_segmentation = self.current_segmentation is not None
        # Show all / Hide all are always present (disabled until a segmentation is loaded) and keep the menu open
        show_all = self.segment_visibility_menu.addAction("Show all levels")
        show_all.triggered.connect(lambda: self.set_all_levels_visible(True))
        hide_all = self.segment_visibility_menu.addAction("Hide all levels")
        hide_all.triggered.connect(lambda: self.set_all_levels_visible(False))
        for action in (show_all, hide_all):
            action.setProperty("keepOpen", True)
            action.setEnabled(has_segmentation)
        self.segment_visibility_menu.addSeparator()
        if has_segmentation:
            for level in self.unique_levels:
                if level != 0:  # Skip background level
                    action = QAction(f"Level {level}", self, checkable=True)
                    action.setChecked(level not in self.invisible_levels)
                    action.triggered.connect(lambda checked, lvl=level: self.toggle_segment_level_visibility(lvl, checked))
                    self.segment_visibility_menu.addAction(action)

    def update_overlap_level_menu(self):
        self._level_menu_key = None
        current_text = self.overlap_level_combo.currentText()  # Get the current selected text
        current_level = self.segmentation_level_spinbox.value()
        levels = [level for level in self.unique_levels if level != 0 and level != current_level]

        self.overlap_level_combo.clear()
        self.overlap_level_combo.addItem("All")
        for level in levels:
            self.overlap_level_combo.addItem(str(level))

        # Check if the previously selected text is still a valid option and set it
        if current_text in [str(level) for level in levels] + ["All"]:
            index = self.overlap_level_combo.findText(current_text)
            self.overlap_level_combo.setCurrentIndex(index)
        else:
            self.overlap_level_combo.setCurrentIndex(0)  # Default to "All" or the first item

    def toggle_overwrite_level(self, level, checked):
        if checked:
            self.overwrite_levels.add(level)
        else:
            self.overwrite_levels.discard(level)

    def update_overwrite_levels_menu(self):
        self._level_menu_key = None
        self.overwrite_levels_menu.clear()
        if self.current_segmentation is not None:
            for level in self.unique_levels:
                action = QAction(f"Level {level}", self, checkable=True)
                action.setChecked(level in self.overwrite_levels)
                action.toggled.connect(partial(self.toggle_overwrite_level, level))
                self.overwrite_levels_menu.addAction(action)

    def init_segmentation_ui(self):
        # Adding segmentation tools to a toolbar
        self.toolbar = self.addToolBar("Segmentation Tools")

        # Setup parameter adjustment area
        self.setup_parameter_adjustment_area()  # Initialize this first

        # Hand button: left-click drag pans the image and never edits the segmentation
        self._last_non_pan_tool = "3d_contour"
        self.pan_action = QAction(make_hand_icon(), "Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setToolTip("Pan: left-click and drag to move the image without editing the segmentation")
        self.pan_action.toggled.connect(self.on_pan_toggled)
        self.toolbar.addAction(self.pan_action)
        self._pan_separator = self.toolbar.addSeparator()  # other front-of-toolbar buttons are inserted before this

        # Label and combobox for selecting volumes
        volume_label = QLabel("Select MRI Volume:")
        self.toolbar.addWidget(volume_label)
        self.volume_combo = QComboBox()
        self.volume_combo.setMinimumContentsLength(18)
        self.volume_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.volume_combo.setMinimumWidth(220)
        self.volume_combo.activated[str].connect(self.switch_volume)
        self.toolbar.addWidget(self.volume_combo)

        # Label and combobox for selecting segmentations
        segmentation_label = QLabel("Select Segmentation:")
        self.toolbar.addWidget(segmentation_label)
        self.segmentation_combo = QComboBox()
        self.segmentation_combo.setMinimumContentsLength(18)
        self.segmentation_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.segmentation_combo.setMinimumWidth(220)
        self.segmentation_combo.addItem("None")  # Default item indicating no segmentation selected
        self.segmentation_combo.addItems(list(self.segmentation_volumes.keys()))
        self.segmentation_combo.activated[str].connect(self.handle_segmentation_selection)
        self.toolbar.addWidget(self.segmentation_combo)

        # Label and dropdown menu for selecting the segmentation tool
        tool_label = QLabel("Select Tool:")
        self.toolbar.addWidget(tool_label)
        self.tool_selector = QComboBox()
        self.tool_selector.addItems(["None"] + [tool.capitalize() for tool in self.segmentation_tools])
        self.tool_selector.activated[str].connect(lambda tool: self.set_segmentation_tool(tool.lower()))
        self.toolbar.addWidget(self.tool_selector)

        # Label and QSpinBox for specifying segmentation level
        level_widget = QWidget()
        level_layout = QHBoxLayout(level_widget)
        level_label = QLabel("Segmentation Level:")
        self.segmentation_level_spinbox = QSpinBox()
        self.segmentation_level_spinbox.setRange(-999, 999)
        self.segmentation_level_spinbox.setValue(1)
        self.segmentation_level_spinbox.valueChanged.connect(self.update_segmentation_level)
        self.segmentation_level_spinbox.valueChanged.connect(self.update_overlap_level_menu)
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.segmentation_level_spinbox)
        self.level_action = QWidgetAction(self.toolbar)
        self.level_action.setDefaultWidget(level_widget)
        self.toolbar.addAction(self.level_action)

        # Create the Overwrite Levels menu
        self.overwrite_levels_menu = PersistentMenu("Overwrite Levels", self)
        self.overwrite_levels_action = QAction("Overwrite Levels", self)
        self.overwrite_levels_action.setMenu(self.overwrite_levels_menu)
        self.toolbar.addAction(self.overwrite_levels_action)
        self.overwrite_levels_action.setVisible(False)  # Hidden by default

        # Create a QWidget for the overlap management with a label
        overlap_widget = QWidget()
        overlap_layout = QHBoxLayout(overlap_widget)
        overlap_label = QLabel("Overlap Management:")
        self.overlap_management_combo = QComboBox()
        self.overlap_management_combo.addItems([
            "Add around pre-existing",
            "Overwrite pre-existing",
            "Erase from all levels",
            "Erase from this level",
            "Segment overlap only"
        ])
        self.overlap_management_combo.activated[str].connect(self.set_new_segmentation_overlap)
        self.overlap_management_combo.activated[str].connect(self.update_overlap_level_visibility)
        self.overlap_management_combo.activated[str].connect(self.update_overwrite_levels_visibility)

        overlap_layout.addWidget(overlap_label)
        overlap_layout.addWidget(self.overlap_management_combo)
        overlap_widget.setLayout(overlap_layout)
        self.overlap_action = QWidgetAction(self.toolbar)
        self.overlap_action.setDefaultWidget(overlap_widget)
        self.toolbar.addAction(self.overlap_action)

        # Label and dropdown menu for selecting the overlap level
        overlap_level_widget = QWidget()
        overlap_level_layout = QHBoxLayout(overlap_level_widget)
        overlap_level_label = QLabel("Overlap Level:")
        self.overlap_level_combo = QComboBox()
        self.overlap_level_combo.addItem("All")  # Default option

        overlap_level_layout.addWidget(overlap_level_label)
        overlap_level_layout.addWidget(self.overlap_level_combo)
        overlap_level_widget.setLayout(overlap_level_layout)
        self.overlap_level_action = QWidgetAction(self.toolbar)
        self.overlap_level_action.setDefaultWidget(overlap_level_widget)
        self.toolbar.addAction(self.overlap_level_action)
        self.overlap_level_action.setVisible(False)

        # Adding label and QLineEdit for level naming at the end
        level_name_widget = QWidget()
        level_name_layout = QHBoxLayout(level_name_widget)
        level_name_label = QLabel("Level Name:")
        self.level_name_edit = QLineEdit()
        self.level_name_edit.setPlaceholderText("Enter level name")
        self.level_name_edit.setMinimumWidth(140)
        self.level_name_edit.setMaximumWidth(180)
        self.level_name_edit.editingFinished.connect(self.update_level_name)
        level_name_layout.addWidget(level_name_label)
        level_name_layout.addWidget(self.level_name_edit)
        level_name_widget.setLayout(level_name_layout)
        self.level_name_action = QWidgetAction(self.toolbar)
        self.level_name_action.setDefaultWidget(level_name_widget)
        self.toolbar.addAction(self.level_name_action)

        # Default tool is Pan, so clicking on a freshly loaded exam never edits anything by accident
        index = self.tool_selector.findText("pan", Qt.MatchFixedString)
        if index >= 0:
            self.tool_selector.setCurrentIndex(index)
            self.set_segmentation_tool("pan")

        # Adding manage segmentation actions to the menu
        manage_menu = QMenu("Manage Segmentations", self)
        create_action = QAction("Create New Segmentation", self)
        create_action.triggered.connect(self.create_new_segmentation_prompt)
        manage_menu.addAction(create_action)

        load_action = QAction("Load Segmentation", self)
        load_action.triggered.connect(self.load_segmentation)
        manage_menu.addAction(load_action)

        unload_action = QAction("Unload Segmentation", self)
        unload_action.triggered.connect(self.delete_segmentation)
        manage_menu.addAction(unload_action)

        rename_action = QAction("Rename Segmentation", self)
        rename_action.triggered.connect(self.rename_segmentation)
        manage_menu.addAction(rename_action)

        clear_action = QAction("Clear Segmentation", self) 
        clear_action.triggered.connect(self.clear_current_segmentation)
        manage_menu.addAction(clear_action)

        save_action = QAction("Save Segmentation", self)
        save_action.triggered.connect(self.save_current_segmentation)
        manage_menu.addAction(save_action)

        quicksave_nrrd_action = QAction("Quicksave to nrrd", self)
        quicksave_nrrd_action.triggered.connect(self.quicksave_to_nrrd)
        manage_menu.addAction(quicksave_nrrd_action)

        quicksave_nifti_action = QAction("Quicksave to nifti", self)
        quicksave_nifti_action.triggered.connect(self.quicksave_to_nifti)
        manage_menu.addAction(quicksave_nifti_action)

        self.menu_bar.addMenu(manage_menu)

        # Adding "Define ROI" menu next to "Manage Segmentations"
        define_roi_menu = QMenu("Define ROI", self)
        
        # Create actions for the menu
        define_roi_action = QAction("Define ROI", self)
        define_roi_action.triggered.connect(self.define_roi)
        define_roi_menu.addAction(define_roi_action)
        
        remove_roi_action = QAction("Remove ROI", self)
        remove_roi_action.triggered.connect(self.remove_roi)
        define_roi_menu.addAction(remove_roi_action)
        
        # Add the "Define ROI" menu to the menu bar
        self.menu_bar.addMenu(define_roi_menu)

        clear_env_menu = QMenu("Clear Environment", self)
        clear_env_action = QAction("Clear Loaded Data", self)
        clear_env_action.triggered.connect(self.clear_environment)
        clear_env_menu.addAction(clear_env_action)
        self.menu_bar.addMenu(clear_env_menu)

    def set_new_segmentation_overlap(self, selection):
        # Map UI selection to internal variable settings
        if selection == "Add around pre-existing":
            self.new_segmentation_overlap = 'add_around'
        elif selection == "Overwrite pre-existing":
            self.new_segmentation_overlap = 'overwrite'
        elif selection == "Erase from all levels":
            self.new_segmentation_overlap = 'erase'
        elif selection == "Erase from this level":
            self.new_segmentation_overlap = 'erase_level'
        elif selection == "Segment overlap only":
            self.new_segmentation_overlap = 'overlap_only'

    def setup_parameter_adjustment_area(self):
        # Base layout for parameter area
        self.parameter_layout = QVBoxLayout(self.parameter_area)

        # 3D Contour Parameters
        self.contour_params_widget = QWidget()
        contour_params_layout = QHBoxLayout(self.contour_params_widget)  # Use QHBoxLayout for the widget's internal layout

        # Widgets for 3D contour parameters
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity Percentage:")
        self.intensity_percentage_line_edit = QLineEdit("0.05")  # Default value
        self.intensity_percentage_line_edit.setValidator(QDoubleValidator(0.0, 1.0, 4))
        intensity_layout.addWidget(intensity_label)
        intensity_layout.addWidget(self.intensity_percentage_line_edit)

        gradient_layout = QHBoxLayout()
        gradient_label = QLabel("Gradient Percentage:")
        self.gradient_percentage_line_edit = QLineEdit("0.01")  # Default value
        self.gradient_percentage_line_edit.setValidator(QDoubleValidator(0.0, 1.0, 4))
        gradient_layout.addWidget(gradient_label)
        gradient_layout.addWidget(self.gradient_percentage_line_edit)

        max_ticks_layout = QHBoxLayout()
        max_ticks_label = QLabel("Max Ticks:")
        self.max_ticks_line_edit = QLineEdit("50000")  # Default value
        self.max_ticks_line_edit.setValidator(QIntValidator(1, 1000000))
        max_ticks_layout.addWidget(max_ticks_label)
        max_ticks_layout.addWidget(self.max_ticks_line_edit)

        # Add rows to the contour parameters layout
        contour_params_layout.addLayout(intensity_layout)
        contour_params_layout.addLayout(gradient_layout)
        contour_params_layout.addLayout(max_ticks_layout)

        # Buttons for default settings
        buttons_layout = QHBoxLayout()
        csf_defaults_button = QPushButton("CSF Default")
        csf_defaults_button.clicked.connect(lambda: self.set_defaults(0.05, 0.01, 50000))
        dim_enhancing_defaults_button = QPushButton("Dim Enhancement Default")
        dim_enhancing_defaults_button.clicked.connect(lambda: self.set_defaults(0.05, 0.02, 50000))
        med_enhancing_defaults_button = QPushButton("Medium Enhancement Default")
        med_enhancing_defaults_button.clicked.connect(lambda: self.set_defaults(0.1, 0.1, 50000))
        bright_enhancing_defaults_button = QPushButton("Bright Enhancement Default")
        bright_enhancing_defaults_button.clicked.connect(lambda: self.set_defaults(0.3, 0.5, 50000))
        buttons_layout.addWidget(csf_defaults_button)
        buttons_layout.addWidget(dim_enhancing_defaults_button)
        buttons_layout.addWidget(med_enhancing_defaults_button)
        buttons_layout.addWidget(bright_enhancing_defaults_button)
        contour_params_layout.addLayout(buttons_layout)

        self.contour_params_widget.setLayout(contour_params_layout)
        self.parameter_layout.addWidget(self.contour_params_widget)
        self.contour_params_widget.setVisible(False)

        # 3D Patch Contour Parameters
        self.patch_contour_params_widget = QWidget()
        patch_contour_layout = QHBoxLayout(self.patch_contour_params_widget)
        
        # Intensity Threshold
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity Threshold (x std):")
        self.intensity_threshold_line_edit = QLineEdit("1.0")  # Default value
        self.intensity_threshold_line_edit.setValidator(QDoubleValidator(0.0, 10.0, 2))
        intensity_layout.addWidget(intensity_label)
        intensity_layout.addWidget(self.intensity_threshold_line_edit)
        
        # Max Iterations
        iterations_layout = QHBoxLayout()
        iterations_label = QLabel("Max Iterations:")
        self.max_iterations_line_edit = QLineEdit("1000000")  # Default value
        self.max_iterations_line_edit.setValidator(QIntValidator(10, 10000000))
        iterations_layout.addWidget(iterations_label)
        iterations_layout.addWidget(self.max_iterations_line_edit)
        
        # Neighborhood Size
        neighborhood_layout = QHBoxLayout()
        neighborhood_label = QLabel("Neighborhood Size:")
        self.neighborhood_size_line_edit = QLineEdit("1")  # Default value
        self.neighborhood_size_line_edit.setValidator(QIntValidator(1, 5))
        neighborhood_layout.addWidget(neighborhood_label)
        neighborhood_layout.addWidget(self.neighborhood_size_line_edit)
        
        # Add layouts to the patch contour parameters widget
        patch_contour_layout.addLayout(intensity_layout)
        patch_contour_layout.addLayout(iterations_layout)
        patch_contour_layout.addLayout(neighborhood_layout)
        
        self.parameter_layout.addWidget(self.patch_contour_params_widget)
        self.patch_contour_params_widget.setVisible(False)  # Hidden by default

        # 2D Brush Parameters
        self.brush_params_widget = QWidget()
        brush_params_layout = QHBoxLayout(self.brush_params_widget)
        brush_size_label = QLabel("Brush Size: 10") #Initializing with default brush size label
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(100)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.valueChanged.connect(lambda value: brush_size_label.setText(f"Brush Size: {value}"))
        self.brush_thickness_slider = QSlider(Qt.Horizontal)
        self.brush_thickness_slider.setMinimum(1)
        self.brush_thickness_slider.setMaximum(200)
        self.brush_thickness_slider.setValue(1)  # Default thickness
        brush_thickness_label = QLabel("Brush Thickness: 1")
        self.brush_thickness_slider.valueChanged.connect(lambda value: brush_thickness_label.setText(f"Brush Thickness: {value}"))
        brush_params_layout.addWidget(brush_size_label)
        brush_params_layout.addWidget(self.brush_size_slider)
        brush_params_layout.addWidget(brush_thickness_label)
        brush_params_layout.addWidget(self.brush_thickness_slider)
        self.brush_position_combo = QComboBox()
        self.brush_position_combo.addItems(["Centered", "Above", "Below"])
        self.brush_position_combo.currentTextChanged.connect(self.update_brush_position)
        brush_params_layout.addWidget(QLabel("Brush Position:"))
        brush_params_layout.addWidget(self.brush_position_combo)
        self.parameter_layout.addWidget(self.brush_params_widget)
        self.brush_params_widget.setVisible(False)

        # Grow Border Parameters
        self.grow_border_params_widget = QWidget()
        grow_border_params_layout = QHBoxLayout(self.grow_border_params_widget)

        # SpinBox for border change
        self.border_change_spinbox = QSpinBox()
        self.border_change_spinbox.setRange(1, 1000)  # Allow shrinking by up to 1000 voxels
        self.border_change_spinbox.setValue(1)  # Default to grow by 1 voxel
        grow_border_params_layout.addWidget(QLabel("# of Voxels to Grow or Shrink Segmentation:"))
        grow_border_params_layout.addWidget(self.border_change_spinbox)

        # Button to grow segmentation
        self.grow_button = QPushButton("Grow Segmentation")
        self.grow_button.clicked.connect(lambda: self.modify_segmentation_borders(self.border_change_spinbox.value(), "grow"))
        grow_border_params_layout.addWidget(self.grow_button)

        # Button to shrink segmentation
        self.shrink_button = QPushButton("Shrink Segmentation")
        self.shrink_button.clicked.connect(lambda: self.modify_segmentation_borders(self.border_change_spinbox.value(), "shrink"))
        grow_border_params_layout.addWidget(self.shrink_button)

        # Button to hollow segmentation
        self.hollow_button = QPushButton("Hollow Segmentation")
        self.hollow_button.clicked.connect(lambda: self.modify_segmentation_borders(self.border_change_spinbox.value(), "hollow"))
        grow_border_params_layout.addWidget(self.hollow_button)

        self.parameter_layout.addWidget(self.grow_border_params_widget)
        self.grow_border_params_widget.setVisible(False)

    def update_overlap_level_visibility(self, text):
        # Show or hide the Overlap Level menu based on the selected text
        if text == "Segment overlap only":
            self.overlap_level_action.setVisible(True)
            self.update_overlap_level_menu()  # Update the content if needed
        else:
            self.overlap_level_action.setVisible(False)

    def update_overwrite_levels_visibility(self, text):
        # Show or hide the Overwrite Levels menu based on the selected text
        if text == "Overwrite pre-existing":
            self.overwrite_levels_action.setVisible(True)
            self.overwrite_levels = set(self.unique_levels)  # Include all levels by default
            self.update_overwrite_levels_menu()
        else:
            self.overwrite_levels_action.setVisible(False)
    
    def update_brush_position(self, position):
        self.brush_position = position

    def set_defaults(self, intensity, gradient, max_ticks):
        self.intensity_percentage_line_edit.setText(f"{intensity:.4f}")
        self.gradient_percentage_line_edit.setText(f"{gradient:.4f}")
        self.max_ticks_line_edit.setText(str(max_ticks))

    def create_parameter_slider(self, label, min_val, max_val, value, layout):
        label_widget = QLabel(f"{label}: {value * 0.01:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(value)
        slider.valueChanged.connect(lambda val, lbl=label_widget: lbl.setText(f"{label}: {val * 0.01:.2f}"))
        layout.addWidget(label_widget)
        layout.addWidget(slider)
        return slider

    # Code to map spatial orientation between nrrd and nifti
    def determine_nrrd_space(self, affine):
        return determine_nrrd_space(affine)

    # Segmentation management functions (load, create, delete, rename, etc.)
    def create_new_segmentation_prompt(self):
        new_name, ok = QInputDialog.getText(self, 'New Segmentation', 'Enter name for new segmentation:')
        if ok and new_name:
            self.create_new_segmentation(new_name)

    def create_new_segmentation(self, name):
        if self.data is None:
            QMessageBox.warning(self, "Create Segmentation", "Load an exam before creating a segmentation.")
            return
        if name in self.segmentation_volumes:
            QMessageBox.warning(self, "Error", "Segmentation already exists!")
        else:
            # Creating new segmentation
            new_segmentation = np.zeros_like(self.data, dtype=int)
            self.segmentation_volumes[name] = new_segmentation
            self.segmentation_names[name] = {}
            self.segmentation_dirty[name] = True
            self.segmentation_combo.addItem(name)
            self.switch_segmentation(name)

    def reorient_affine_space_to_ras(self, affine, current_space):
        return reorient_affine_space_to_ras(affine, current_space)

    def load_segmentation(self, file_path=None):
        if self.data is None:
            QMessageBox.warning(self, "Load Segmentation", "Load an exam before loading a segmentation.")
            return

        # QAction.triggered emits a boolean "checked" argument.
        # Treat bool exactly like no explicit file path so the file dialog opens.
        if file_path is None or isinstance(file_path, bool):
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Open Segmentation File(s)",
                self.default_save_directory(),
                "Segmentation Files (*.nii *.nii.gz *.nrrd)"
            )
        elif isinstance(file_path, (list, tuple)):
            file_paths = list(file_path)
        else:
            file_paths = [file_path]

        if not file_paths:
            return

        loaded_paths = []
        duplicate_names = []
        incompatible_names = []
        failed_files = []

        for seg_path in file_paths:
            if not seg_path:
                continue
            try:
                data, level_names = load_segmentation_data(seg_path)
                seg_name = os.path.basename(seg_path)

                if data.shape != self.master_canonical_shape:
                    incompatible_names.append(
                        f"{seg_name} (shape {data.shape}, expected {self.master_canonical_shape})"
                    )
                    continue

                if seg_name in self.segmentation_volumes:
                    duplicate_names.append(seg_name)
                    continue

                self.segmentation_volumes[seg_name] = data
                self.segmentation_names[seg_name] = level_names
                self.segmentation_dirty[seg_name] = False
                self.segmentation_combo.addItem(seg_name)
                loaded_paths.append(seg_path)
            except Exception as e:
                failed_files.append(f"{os.path.basename(seg_path)}: {str(e)}")

        if loaded_paths:
            last_seg_name = os.path.basename(loaded_paths[-1])
            self.segmentation_combo.setCurrentText(last_seg_name)
            self.current_segmentation = self.segmentation_volumes[last_seg_name]
            self.current_segmentation_key = last_seg_name
            self.unique_levels = np.unique(self.current_segmentation)
            self.update_overlap_level_menu()
            self.update_level_name_display()
            self.update_segment_visibility_menu()
            self.update_overwrite_levels_menu()
            self.display_all_slices()
            self.update_visualization(reset_volume=False)
            self.display_color_key_on_3d()
            self.segmentation_paths = loaded_paths

        messages = []
        if duplicate_names:
            messages.append("Already loaded: " + ", ".join(duplicate_names))
        if incompatible_names:
            messages.append("Incompatible shape: " + "; ".join(incompatible_names))
        if failed_files:
            messages.append("Failed to load: " + "; ".join(failed_files))
        if messages:
            QMessageBox.information(self, "Load Segmentation", "\n\n".join(messages))

    def delete_segmentation(self):
        if self.current_segmentation_key:
            reply = QMessageBox.question(self, 'Confirm Unload',
                                        f"Unload '{self.current_segmentation_key}' from the viewer? This will not delete anything from disk.",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                del self.segmentation_volumes[self.current_segmentation_key]
                self.segmentation_names.pop(self.current_segmentation_key, None)
                self.segmentation_dirty.pop(self.current_segmentation_key, None)
                self.segmentation_combo.removeItem(self.segmentation_combo.findText(self.current_segmentation_key))
                self.switch_segmentation(None)

    def rename_segmentation(self):
        if self.current_segmentation_key:
            new_name, ok = QInputDialog.getText(self, 'Rename Segmentation',
                                                'Enter new name for the segmentation:',
                                                text=self.current_segmentation_key)
            if ok and new_name:
                old_name = self.current_segmentation_key
                self.segmentation_volumes[new_name] = self.segmentation_volumes.pop(old_name)
                self.segmentation_names[new_name] = self.segmentation_names.pop(old_name, {})
                self.segmentation_dirty[new_name] = self.segmentation_dirty.pop(old_name, False)
                index = self.segmentation_combo.findText(old_name)
                self.segmentation_combo.setItemText(index, new_name)
                self.current_segmentation_key = new_name
                self.mark_segmentation_dirty(new_name, True)

    def clear_current_segmentation(self):
        if self.current_segmentation_key:
            reply = QMessageBox.question(self, 'Confirm Clear',
                                         f"Are you sure you want to clear all data in '{self.current_segmentation_key}'?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.current_segmentation.fill(0)  # Set all values in the current segmentation to 0
                self.unique_levels = np.unique(self.current_segmentation)
                self.mark_segmentation_dirty(self.current_segmentation_key, True)
                self.update_overlap_level_menu()
                self.update_segment_visibility_menu()
                self.update_overwrite_levels_menu()
                self.update_visualization(reset_volume=False)
                self.display_all_slices()
        else:
            QMessageBox.warning(self, "No Segmentation Selected", "Please select a segmentation volume to clear.")

    def load_roi(self, file_path):
        if self.data is None:
            QMessageBox.warning(self, "Define ROI", "Load an exam before loading an ROI mask.")
            return
        # Load the segmentation file using the existing load_and_reorient_to_RAS method
        segmentation_data, segmentation_affine, voxel_dims = self.load_and_reorient_to_RAS(file_path)
        
        if segmentation_data is not None:
            # Perform size and orientation checks
            if self.check_segmentation_compatibility(segmentation_data, segmentation_affine):
                # Create ROI mask where segmentation is non-zero
                self.roi_mask = segmentation_data > 0 
                # Updating ROI Loaded label in menu bar               
                filename = os.path.basename(file_path)
                self.roi_label.setText(f"ROI Loaded: {filename}")
            else:
                QMessageBox.warning(
                    self,
                    "Define ROI",
                    "The loaded segmentation does not match the size or orientation of the loaded MRI data."
                )
        else:
            QMessageBox.warning(
                self,
                "Define ROI",
                "Failed to load the segmentation file."
            )

    def define_roi(self):
        """
        Opens a file dialog to load a segmentation file and defines the ROI mask based on it.
        Ensures that the loaded segmentation matches the size and orientation of the loaded MRI data.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Define ROI from Segmentation",
            self.default_save_directory(),
            "Segmentation Files (*.nii *.nii.gz *.nrrd)"
        )
        
        if file_path:
            try:
                self.load_roi(file_path)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Define ROI Error",
                    f"An error occurred while defining ROI: {str(e)}"
                )

    def remove_roi(self):
        """
        Removes the current ROI mask.
        """
        if self.roi_mask is not None:
            self.roi_mask = None
            # Update the ROI Loaded Label in the Menu Bar
            self.roi_label.setText("ROI Loaded: None")
        else:
            QMessageBox.information(
                self,
                "Remove ROI",
                "No ROI is currently defined."
            )
    
    def check_segmentation_compatibility(self, segmentation_data, segmentation_affine):
        """
        Checks whether the loaded segmentation data matches the loaded MRI data in size and orientation.
        
        Parameters:
            segmentation_data (np.array): The segmentation data loaded from file.
            segmentation_affine (np.array): The affine matrix of the segmentation data.
        
        Returns:
            bool: True if compatible, False otherwise.
        """
        # Check if self.data exists
        if self.data is None:
            print("Error: MRI data (self.data) is not loaded.")
            return False
        
        # Check shape
        if segmentation_data.shape != self.data.shape:
            print(f"Shape mismatch: Segmentation shape {segmentation_data.shape} vs MRI data shape {self.data.shape}")
            return False
        
        # Check orientation by comparing affines using nibabel
        try:
            seg_ornt = nib.orientations.io_orientation(segmentation_affine)
            mri_ornt = nib.orientations.io_orientation(self.affine)  # Ensure self.affine is correctly set
            if not np.array_equal(seg_ornt, mri_ornt):
                print("Orientation mismatch between segmentation and MRI data.")
                return False
        except Exception as e:
            print(f"Error during orientation check: {e}")
            return False
        
        # If all checks pass
        return True

    def on_pan_toggled(self, checked):
        if checked:
            self.set_segmentation_tool("pan")
        else:
            # Clicking the hand again goes back to the tool that was active before
            self.set_segmentation_tool(self._last_non_pan_tool)

    def set_segmentation_tool(self, tool):
        # Save the current Overlap Management selection
        current_overlap_selection = self.overlap_management_combo.currentText()

        # Reset default visibility for all parameter widgets
        self.contour_params_widget.setVisible(False)
        self.patch_contour_params_widget.setVisible(False)
        self.brush_params_widget.setVisible(False)
        self.grow_border_params_widget.setVisible(False)
        self.adjust_overlap_options_for_grow_borders(False)
        self.level_action.setVisible(True)
        self.overlap_action.setVisible(True)
        # Reset visibilty for mouse highlights that may be specific to a tool
        self.remove_mouse_highlights()

        # Update current tool
        self.current_segmentation_tool = tool.lower() if tool != "None" else None

        # Show relevant parameter widgets based on the selected tool
        if self.current_segmentation_tool in ('3d_contour', '3d_contour_global'):
            self.contour_params_widget.setVisible(True)
        elif self.current_segmentation_tool == "3d_patch_contour":
            self.patch_contour_params_widget.setVisible(True)
            self.brush_params_widget.setVisible(True)
        elif self.current_segmentation_tool == '2d_brush':
            self.brush_params_widget.setVisible(True)
        elif self.current_segmentation_tool == 'grow_borders':
            self.grow_border_params_widget.setVisible(True)
            self.adjust_overlap_options_for_grow_borders(True)
        elif self.current_segmentation_tool in ("remove_volume", "keep_volume_only", "fill_holes"):
            self.level_action.setVisible(False)
            self.overlap_action.setVisible(False)
        elif self.current_segmentation_tool in ("change_volume_level"):
            self.overlap_action.setVisible(False)

        # Keep the hand button, the tool dropdown and the view cursors in sync with the active tool
        is_pan = self.current_segmentation_tool == 'pan'
        if self.current_segmentation_tool and not is_pan:
            self._last_non_pan_tool = self.current_segmentation_tool
        self.pan_action.blockSignals(True)
        self.pan_action.setChecked(is_pan)
        self.pan_action.blockSignals(False)
        tool_index = self.tool_selector.findText(self.current_segmentation_tool or "None", Qt.MatchFixedString)
        if tool_index >= 0:
            self.tool_selector.setCurrentIndex(tool_index)
        for view in self.views.values():
            if is_pan:
                view.viewport().setCursor(Qt.OpenHandCursor)
            else:
                view.viewport().unsetCursor()

        # Restore the Overlap Management selection
        if self.overlap_action.isVisible():
            index = self.overlap_management_combo.findText(current_overlap_selection)
            if index >= 0:
                self.overlap_management_combo.setCurrentIndex(index)
            else:
                self.overlap_management_combo.setCurrentIndex(0)  # Default to the first option

    def adjust_overlap_options_for_grow_borders(self, active):
        if active:
            # Remove options not applicable for grow_borders
            self.overlap_management_combo.clear()
            self.overlap_management_combo.addItems([
                "Add around pre-existing",
                "Overwrite pre-existing"
            ])
            # Set default
            self.overlap_management_combo.setCurrentIndex(0)
        else:
            # Restore all options
            self.overlap_management_combo.clear()
            self.overlap_management_combo.addItems([
                "Add around pre-existing",
                "Overwrite pre-existing",
                "Erase from all levels",
                "Erase from this level",
                "Segment overlap only"
            ])

    def handle_segmentation_selection(self, text):
        if text == "None":
            self.current_segmentation = None
            self.current_segmentation_key = None
            self.update_visualization(reset_volume=False)
            self.display_all_slices()
        elif text == "Create new segmentation":
            new_name, ok = QInputDialog.getText(self, 'New Segmentation', 'Enter name for new segmentation:')
            if ok and new_name:
                self.create_new_segmentation(new_name)
        else:
            self.switch_segmentation(text)

    def update_level_name(self):
        if self.current_segmentation_key:
            level = self.segmentation_level_spinbox.value()
            name = self.level_name_edit.text()
            if name:  # Only update if there's actually a name entered
                self.segmentation_names[self.current_segmentation_key][level] = name
                self.display_color_key_on_3d()

    def update_level_name_display(self):
        if self.current_segmentation_key:
            level = self.segmentation_level_spinbox.value()
            # Fetch the name from the dictionary, defaulting to an empty string if not found
            name = self.segmentation_names.get(self.current_segmentation_key, {}).get(level, "")
            self.level_name_edit.setText(name)
        else:
            self.level_name_edit.setText("")

    def switch_segmentation(self, name):
        if name is None or name == "None":
            self.current_segmentation = None
            self.current_segmentation_key = None
            self.unique_levels = np.array([0])
            self.segmentation_combo.setCurrentText("None")
        else:
            self.current_segmentation = self.segmentation_volumes.get(name)
            self.current_segmentation_key = name
            self.unique_levels = np.unique(self.current_segmentation) if self.current_segmentation is not None else np.array([0])
            self.segmentation_combo.setCurrentText(name)
        self.update_level_name_display()
        self.update_segment_visibility_menu()
        self.update_overwrite_levels_menu()
        self.update_overlap_level_menu()
        self.update_visualization(reset_volume=False)  # Refreshes the 3D visualization
        self.display_all_slices()  # Refreshes the 2D slice views

    def add_toolbar_actions(self):
        brush_action = QAction("Brush", self)
        brush_action.triggered.connect(lambda: self.set_segmentation_tool("brush"))
        self.toolbar.addAction(brush_action)

        erase_action = QAction("Erase", self)
        erase_action.triggered.connect(lambda: self.set_segmentation_tool("erase"))
        self.toolbar.addAction(erase_action)

    def manage_segmentations(self):
        # Optional: Dialog or method to manage existing segmentations (rename, delete, etc.)
        pass

    def load_and_reorient_to_RAS(self, file_path):
        try:
            loaded = load_and_reorient_volume(
                file_path,
                master_affine=self.master_affine,
                master_orientation=self.master_orientation,
                master_voxel_dims=self.master_voxel_dims,
                master_shape=self.master_shape,
                master_canonical_affine=self.master_canonical_affine,
                master_canonical_orientation=self.master_canonical_orientation,
                master_canonical_voxel_dims=self.master_canonical_voxel_dims,
                master_canonical_shape=self.master_canonical_shape,
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))
            return None, None, None

        img = nib.load(file_path)
        if self.master_affine is None:
            self.master_affine = img.affine
        if self.master_orientation is None:
            self.master_orientation = self.determine_nrrd_space(self.master_affine)
        if self.master_voxel_dims is None:
            self.master_voxel_dims = img.header.get_zooms()
        if self.master_shape is None:
            self.master_shape = img.get_fdata().shape
        if self.master_canonical_affine is None:
            self.master_canonical_affine = loaded.affine
        if self.master_canonical_orientation is None:
            self.master_canonical_orientation = self.determine_nrrd_space(loaded.affine)
        if self.master_canonical_voxel_dims is None:
            self.master_canonical_voxel_dims = loaded.voxel_dims
        if self.master_canonical_shape is None:
            self.master_canonical_shape = loaded.data.shape

        return loaded.data, loaded.affine, loaded.voxel_dims

    def reorient_to_original(self, data):
        return reorient_volume_to_original(
            data,
            master_canonical_affine=self.master_canonical_affine,
            master_affine=self.master_affine,
        )










            






    
    

























    def open_rotation_dialog(self):
        dialog = RotationDialog(self)
        dialog.exec_()

    def update_rotation(self, angle, axis):
        if self.data is None:
            return
        order = 1  # Typically interactive update
        if axis == 'x':
            self.data = rotate(self.data, angle, axes=(1, 2), reshape=False, order=order)
        elif axis == 'y':
            self.data = rotate(self.data, angle, axes=(0, 2), reshape=False, order=order)
        elif axis == 'z':
            self.data = rotate(self.data, angle, axes=(0, 1), reshape=False, order=order)

        # Refresh the 3D volume in the plotter
        self.create_3d_volume(self.data, self.voxel_dims)

        # Re-create the 3D planes based on the new data orientation
        self.update_planes()

        # Refresh the 2D slice views
        self.display_all_slices()

    def reset_rotation(self):
        if self.original_data is None:
            return
        self.data = np.copy(self.original_data)
        self.display_all_slices()








        
