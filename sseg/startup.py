from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QWidget, QApplication,
    QFormLayout, QGroupBox, QTabWidget, QPlainTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QComboBox, QProgressDialog, QScrollArea
)

from .batch_discovery import (
    analyze_exam,
    discover_exams,
    exam_has_all_selected_series,
    get_available_series_types_for_suffix,
    count_exams_with_suffix,
    get_detected_suffixes,
)
from .batch_io import load_batch_discovery, save_batch_discovery
from .batch_models import (
    BatchDiscoveryResult,
    BatchDiscoverySettings,
    BatchLaunchSelection,
    DEFAULT_BRAINMASK_SUFFIXES,
    DEFAULT_SEGMENTATION_SUFFIXES,
    DEFAULT_VOLUME_SUFFIXES,
    DEFAULT_IMAGE_EXTENSIONS,
)


@dataclass
class StartupConfig:
    volume_paths: List[str]
    segmentation_paths: List[str]
    segmentation_suffixes: List[str]
    suffix_to_replace: Optional[str]
    roi_mask_path: Optional[str]
    batch_discovery_result: Optional[BatchDiscoveryResult] = None
    batch_launch_selection: Optional[BatchLaunchSelection] = None


class BatchDiscoveryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.discovery_result: Optional[BatchDiscoveryResult] = None

        root = QVBoxLayout(self)
        intro = QLabel(
            'Scan recursively for astril-style exam folders. Exam folders are expected to be named '
            '{patient_id}_{timepoint}_{exam_id}, to live directly under the patient folder, and to '
            'contain top-level MRI series volumes that start with the patient id. This tab also lets '
            'you choose a volume type and a set of series types for the next batch step.'
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        path_group = QGroupBox('Directory to scan')
        path_layout = QHBoxLayout(path_group)
        self.root_dir_edit = QLineEdit()
        self.root_dir_edit.setPlaceholderText('Select a root folder above, at, or below the patient level')
        path_layout.addWidget(self.root_dir_edit)
        browse_btn = QPushButton('Browse…')
        browse_btn.clicked.connect(self.select_root_dir)
        path_layout.addWidget(browse_btn)
        scan_btn = QPushButton('Scan Directory for Exams')
        scan_btn.clicked.connect(self.scan_directory)
        path_layout.addWidget(scan_btn)
        root.addWidget(path_group)

        adv_group = QGroupBox('Advanced discovery settings')
        adv_form = QFormLayout(adv_group)
        self.extensions_edit = QLineEdit(' '.join(DEFAULT_IMAGE_EXTENSIONS))
        self.extensions_edit.setPlaceholderText('.nii .nii.gz .nrrd')
        adv_form.addRow('Image extensions:', self.extensions_edit)
        self.brainmask_edit = QPlainTextEdit('\n'.join(DEFAULT_BRAINMASK_SUFFIXES))
        self.brainmask_edit.setPlaceholderText('One brainmask suffix per line')
        self.brainmask_edit.setFixedHeight(90)
        adv_form.addRow('Brainmask suffixes:', self.brainmask_edit)
        self.seg_suffix_edit = QPlainTextEdit('\n'.join(DEFAULT_SEGMENTATION_SUFFIXES))
        self.seg_suffix_edit.setPlaceholderText('One segmentation suffix per line')
        self.seg_suffix_edit.setFixedHeight(90)
        adv_form.addRow('Segmentation suffixes:', self.seg_suffix_edit)
        self.volume_suffix_edit = QPlainTextEdit('\n'.join(DEFAULT_VOLUME_SUFFIXES))
        self.volume_suffix_edit.setPlaceholderText(
            'One volume suffix per line, including extension, e.g. _brain-norm.nii.gz.\n'
            'The text before the suffix, after its last "_", is the series type (T1c, FLAIR, ...).'
        )
        self.volume_suffix_edit.setFixedHeight(70)
        adv_form.addRow('Volume suffixes:', self.volume_suffix_edit)
        self.verify_compat_checkbox = QCheckBox('Verify Scan Compatibility (slower)')
        self.verify_compat_checkbox.setChecked(False)
        adv_form.addRow('', self.verify_compat_checkbox)
        root.addWidget(adv_group)

        planning_group = QGroupBox('Batch planning')
        planning_form = QFormLayout(planning_group)
        self.suffix_combo = QComboBox()
        self.suffix_combo.setEditable(False)
        self.suffix_combo.currentTextChanged.connect(self.on_suffix_changed)
        planning_form.addRow('Selected volume type:', self.suffix_combo)
        self.series_list = QListWidget()
        self.series_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.series_list.itemSelectionChanged.connect(self.on_series_selection_changed)
        self.series_list.setMinimumHeight(90)
        planning_form.addRow('Series types to load:', self.series_list)
        self.require_all_checkbox = QCheckBox('Only display exams that contain all selected series')
        self.require_all_checkbox.toggled.connect(self.refresh_results_table)
        planning_form.addRow('', self.require_all_checkbox)
        self.new_seg_suffix_edit = QLineEdit()
        self.new_seg_suffix_edit.setPlaceholderText('Optional new segmentation suffix, e.g. _tumor-segmentation-v2.nii.gz')
        self.new_seg_suffix_edit.textChanged.connect(self.on_new_seg_suffix_changed)
        planning_form.addRow('New segmentation suffix:', self.new_seg_suffix_edit)
        root.addWidget(planning_group)

        results_group = QGroupBox('Discovered exams')
        results_layout = QVBoxLayout(results_group)
        self.summary_label = QLabel('No discovery has been run yet.')
        results_layout.addWidget(self.summary_label)
        self.results_table = QTableWidget(0, 9)
        self.results_table.setMinimumHeight(180)
        self.results_table.setHorizontalHeaderLabels([
            'Patient', 'Exam', 'Exam Directory', 'Series Volumes', 'Volume Types',
            'Series Types', 'Brainmask', 'Segmentation', 'Subdirs'
        ])
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.verticalHeader().setVisible(False)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        for idx in (3, 4, 5, 6, 7, 8):
            header.setSectionResizeMode(idx, QHeaderView.ResizeToContents)
        results_layout.addWidget(self.results_table)

        buttons = QHBoxLayout()
        load_btn = QPushButton('Load Discovery JSON…')
        load_btn.clicked.connect(self.load_discovery_json)
        buttons.addWidget(load_btn)
        save_btn = QPushButton('Save Discovery JSON…')
        save_btn.clicked.connect(self.save_discovery_json)
        buttons.addWidget(save_btn)
        buttons.addStretch(1)
        root.addWidget(results_group)
        root.addLayout(buttons)

    def select_root_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, 'Select directory to scan', self.root_dir_edit.text().strip() or '')
        if path:
            self.root_dir_edit.setText(str(Path(path)))

    def _nonempty_lines(self, text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _build_settings(self) -> BatchDiscoverySettings:
        exts = [item.strip() for item in self.extensions_edit.text().split() if item.strip()]
        if not exts:
            exts = list(DEFAULT_IMAGE_EXTENSIONS)
        brainmask_suffixes = self._nonempty_lines(self.brainmask_edit.toPlainText()) or list(DEFAULT_BRAINMASK_SUFFIXES)
        segmentation_suffixes = self._nonempty_lines(self.seg_suffix_edit.toPlainText()) or list(DEFAULT_SEGMENTATION_SUFFIXES)
        return BatchDiscoverySettings(
            root_dir=self.root_dir_edit.text().strip(),
            image_extensions=exts,
            brainmask_suffixes=brainmask_suffixes,
            segmentation_suffixes=segmentation_suffixes,
            verify_scan_compatibility=bool(self.verify_compat_checkbox.isChecked()),
            volume_suffixes=self._nonempty_lines(self.volume_suffix_edit.toPlainText()) or list(DEFAULT_VOLUME_SUFFIXES),
        )

    def scan_directory(self) -> None:
        settings = self._build_settings()
        if not settings.root_dir:
            QMessageBox.warning(self, 'Batch Discovery', 'Please select a root directory to scan.')
            return

        progress = QProgressDialog('Scanning directories...', 'Cancel', 0, 100, self)
        progress.setWindowTitle('Batch Discovery')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_progress(current: int, total: int, label: str) -> None:
            if total <= 0:
                total = 1
            progress.setMaximum(total)
            progress.setValue(min(current, total))
            progress.setLabelText(label)
            QApplication.processEvents()

        try:
            result = discover_exams(settings.root_dir, settings, progress_callback=on_progress)
        finally:
            progress.close()

        self.set_discovery_result(result)
        if not result.exams:
            QMessageBox.information(
                self,
                'Batch Discovery',
                'No exam folders were discovered with the current settings.'
            )

    def _update_selection_controls(self) -> None:
        if self.discovery_result is None:
            self.suffix_combo.blockSignals(True)
            self.suffix_combo.clear()
            self.suffix_combo.blockSignals(False)
            self.series_list.clear()
            self.require_all_checkbox.setChecked(False)
            self.new_seg_suffix_edit.setText('')
            return
        sel = self.discovery_result.launch_selection
        detected_suffixes = get_detected_suffixes(self.discovery_result)
        if sel.selected_suffix not in detected_suffixes:
            sel.selected_suffix = detected_suffixes[0] if detected_suffixes else None

        self.suffix_combo.blockSignals(True)
        self.suffix_combo.clear()
        for suffix in detected_suffixes:
            self.suffix_combo.addItem(suffix)
        current_idx = self.suffix_combo.findText(sel.selected_suffix or '')
        if current_idx < 0 and (sel.selected_suffix or '').strip():
            self.suffix_combo.addItem(sel.selected_suffix)
            current_idx = self.suffix_combo.findText(sel.selected_suffix)
        self.suffix_combo.setCurrentIndex(max(current_idx, 0))
        self.suffix_combo.blockSignals(False)

        self.require_all_checkbox.blockSignals(True)
        self.require_all_checkbox.setChecked(bool(sel.require_all_selected_series))
        self.require_all_checkbox.blockSignals(False)
        self.new_seg_suffix_edit.blockSignals(True)
        self.new_seg_suffix_edit.setText(sel.replacement_segmentation_suffix or '')
        self.new_seg_suffix_edit.blockSignals(False)
        self._populate_series_list()

    def _populate_series_list(self) -> None:
        self.series_list.clear()
        if self.discovery_result is None:
            return
        suffix = (self.suffix_combo.currentText() or '').strip()
        if not suffix:
            return
        available = get_available_series_types_for_suffix(self.discovery_result, suffix)
        selected = set(self.discovery_result.launch_selection.selected_series_types)
        for series in available:
            item = QListWidgetItem(series)
            self.series_list.addItem(item)
            if series in selected:
                item.setSelected(True)

    def on_suffix_changed(self, _text: str = '') -> None:
        if self.discovery_result is None:
            return
        self.discovery_result.launch_selection.selected_suffix = self.suffix_combo.currentText().strip() or None
        # keep only still-valid series selections
        valid = set(get_available_series_types_for_suffix(self.discovery_result, self.discovery_result.launch_selection.selected_suffix or ''))
        self.discovery_result.launch_selection.selected_series_types = [s for s in self.discovery_result.launch_selection.selected_series_types if s in valid]
        self._populate_series_list()
        self.refresh_results_table()

    def on_series_selection_changed(self) -> None:
        if self.discovery_result is None:
            return
        self.discovery_result.launch_selection.selected_series_types = [item.text() for item in self.series_list.selectedItems()]
        self.refresh_results_table()

    def on_new_seg_suffix_changed(self) -> None:
        if self.discovery_result is None:
            return
        self.discovery_result.launch_selection.replacement_segmentation_suffix = self.new_seg_suffix_edit.text().strip()

    def set_discovery_result(self, result: BatchDiscoveryResult) -> None:
        self.discovery_result = result
        self.root_dir_edit.setText(result.settings.root_dir)
        self.extensions_edit.setText(' '.join(result.settings.image_extensions))
        self.brainmask_edit.setPlainText('\n'.join(result.settings.brainmask_suffixes))
        self.seg_suffix_edit.setPlainText('\n'.join(result.settings.segmentation_suffixes))
        self.verify_compat_checkbox.setChecked(bool(getattr(result.settings, 'verify_scan_compatibility', False)))
        self.volume_suffix_edit.setPlainText('\n'.join(result.settings.volume_suffixes))

        # Re-analyze in case settings or persisted content changed.
        for exam in result.exams:
            analyze_exam(exam, result.settings)

        self._update_selection_controls()
        self.refresh_results_table()

    def refresh_results_table(self) -> None:
        result = self.discovery_result
        self.results_table.setRowCount(0)
        if result is None:
            self.summary_label.setText('No discovery has been run yet.')
            return

        suffix = result.launch_selection.selected_suffix or ''
        selected_series = list(result.launch_selection.selected_series_types)
        require_all = bool(self.require_all_checkbox.isChecked())
        result.launch_selection.require_all_selected_series = require_all

        rows = []
        for exam in result.exams:
            if suffix and suffix not in exam.available_suffixes:
                continue
            if require_all and suffix and not exam_has_all_selected_series(exam, suffix, selected_series):
                continue
            suffixes_display = ', '.join(exam.available_suffixes)
            if suffix:
                series_display = ', '.join(exam.available_series_types_by_suffix.get(suffix, []))
            else:
                all_series = sorted({sf.series_type for sf in exam.series_files})
                series_display = ', '.join(all_series)
            try:
                exam_dir_display = str(Path(exam.exam_dir).resolve().relative_to(Path(result.settings.root_dir).resolve()))
            except Exception:
                exam_dir_display = exam.exam_dir
            rows.append((exam, [
                exam.patient_id,
                exam.exam_id,
                exam_dir_display,
                str(len(exam.series_volume_paths)),
                suffixes_display,
                series_display,
                Path(exam.brainmask_path).name if exam.brainmask_path else '',
                Path(exam.preferred_segmentation_path).name if exam.preferred_segmentation_path else '',
                ', '.join(exam.subdirs_present),
            ]))

        self.results_table.setRowCount(len(rows))
        for row, (exam_obj, values) in enumerate(rows):
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col == 2:
                    item.setToolTip(exam_obj.exam_dir)
                elif col == 6 and exam_obj.brainmask_path:
                    item.setToolTip(exam_obj.brainmask_path)
                elif col == 7 and exam_obj.preferred_segmentation_path:
                    item.setToolTip(exam_obj.preferred_segmentation_path)
                if col == 3:
                    item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, col, item)

        exam_count = len(result.exams)
        visible_count = len(rows)
        skipped_count = len(result.skipped_dirs)
        detected = [f'{sfx} ({count_exams_with_suffix(result, sfx)} exams)' for sfx in get_detected_suffixes(result)]
        self.summary_label.setText(
            f'Discovered {exam_count} exam folder(s), showing {visible_count}. '
            f'Checked {exam_count + skipped_count} directorie(s). Volume types: {", ".join(detected) if detected else "none"}.'
        )
        self.results_table.resizeRowsToContents()

    def load_discovery_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load discovery JSON',
            self.root_dir_edit.text().strip() or '',
            'JSON Files (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            result = load_batch_discovery(path)
        except Exception as exc:
            QMessageBox.critical(self, 'Load Discovery JSON', f'Failed to load discovery JSON\n{exc}')
            return
        self.set_discovery_result(result)

    def save_discovery_json(self) -> None:
        if self.discovery_result is None:
            QMessageBox.warning(self, 'Save Discovery JSON', 'There is no discovery result to save yet.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save discovery JSON',
            str(Path(self.discovery_result.settings.root_dir or '.') / 'sseg_batch_discovery.json'),
            'JSON Files (*.json);;All files (*)',
        )
        if not path:
            return
        try:
            save_batch_discovery(self.discovery_result, path)
        except Exception as exc:
            QMessageBox.critical(self, 'Save Discovery JSON', f'Failed to save discovery JSON:\n{exc}')
            return
        QMessageBox.information(self, 'Save Discovery JSON', f'Saved discovery JSON to:\n{path}')


class StartupDialog(QDialog):
    """Startup workflow for package-based GUI launch.

    This dialog replaces the old requirement that users must launch the viewer
    with ``--load_volumes`` from the command line.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('sseg startup')
        self.setSizeGripEnabled(True)
        self._launch_mode = 'manual'

        root = QVBoxLayout(self)
        # Wrap the startup UI in a scroll area so the full dialog remains reachable
        # on laptops or other small displays, even when the layout's preferred size
        # is larger than the available screen geometry.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        intro = QLabel(
            'Start a manual session, or use batch discovery to find astril-style patient/exam folders '
            'and launch the viewer in empty mode with discovery data attached.'
        )
        intro.setWordWrap(True)
        content_layout.addWidget(intro)

        self.tabs = QTabWidget()
        self.manual_tab = QWidget()
        self.batch_tab = BatchDiscoveryTab()
        self.tabs.addTab(self.manual_tab, 'Manual Session')
        self.tabs.addTab(self.batch_tab, 'Batch Discovery')
        self.tabs.currentChanged.connect(self._update_launch_button_text)
        content_layout.addWidget(self.tabs)

        self._build_manual_tab()

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.launch_button = QPushButton('Launch viewer')
        self.launch_button.clicked.connect(self.validate_and_accept)
        self.launch_button.setDefault(True)
        buttons.addWidget(self.launch_button)
        content_layout.addLayout(buttons)
        self._update_launch_button_text()
        self._resize_for_screen()

    def _resize_for_screen(self) -> None:
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(980, 780)
            return

        available = screen.availableGeometry()
        width = min(980, max(720, available.width() - 80))
        height = min(780, max(520, available.height() - 80))
        self.resize(width, height)

        # Keep the dialog minimum size modest so users can shrink it further
        # if they want to while relying on scrollbars for the overflow content.
        self.setMinimumSize(min(520, width), min(420, height))

    def _build_manual_tab(self) -> None:
        root = QVBoxLayout(self.manual_tab)
        intro = QLabel(
            'Open one or more MRI volumes to start a session, or leave the lists empty to launch an empty viewer. '
            'You can also preload existing segmentations, an ROI mask, and default output naming options.'
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        root.addWidget(self._build_list_group(
            'MRI volumes (.nii, .nii.gz, .nrrd)',
            attr_name='volume_list',
            add_handler=self.add_volumes,
            clear_handler=self.clear_volumes,
        ))
        root.addWidget(self._build_list_group(
            'Segmentations (.nii, .nii.gz, .nrrd)',
            attr_name='segmentation_list',
            add_handler=self.add_segmentations,
            clear_handler=self.clear_segmentations,
        ))

        roi_group = QGroupBox('Optional ROI mask')
        roi_layout = QHBoxLayout(roi_group)
        self.roi_edit = QLineEdit()
        self.roi_edit.setPlaceholderText('No ROI mask selected')
        roi_layout.addWidget(self.roi_edit)
        btn_roi = QPushButton('Browse…')
        btn_roi.clicked.connect(self.select_roi)
        roi_layout.addWidget(btn_roi)
        btn_clear_roi = QPushButton('Clear')
        btn_clear_roi.clicked.connect(lambda: self.roi_edit.setText(''))
        roi_layout.addWidget(btn_clear_roi)
        root.addWidget(roi_group)

        naming_group = QGroupBox('Optional default naming for new segmentations')
        form = QFormLayout(naming_group)
        self.suffixes_edit = QLineEdit()
        self.suffixes_edit.setPlaceholderText('Example: _tumor_seg _edema_seg')
        form.addRow('New segmentation suffixes:', self.suffixes_edit)
        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText('Text to replace in first loaded volume name')
        form.addRow('Suffix to replace:', self.replace_edit)
        root.addWidget(naming_group)
        root.addStretch(1)

    def _update_launch_button_text(self) -> None:
        if self.tabs.currentWidget() is self.batch_tab:
            self.launch_button.setText('Begin Segmenting')
        else:
            self.launch_button.setText('Launch viewer')

    def _build_list_group(self, title: str, attr_name: str, add_handler, clear_handler) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        list_widget = QListWidget()
        setattr(self, attr_name, list_widget)
        layout.addWidget(list_widget)
        row = QHBoxLayout()
        add_btn = QPushButton('Add…')
        add_btn.clicked.connect(add_handler)
        row.addWidget(add_btn)
        remove_btn = QPushButton('Remove selected')
        remove_btn.clicked.connect(lambda: self._remove_selected(list_widget))
        row.addWidget(remove_btn)
        clear_btn = QPushButton('Clear all')
        clear_btn.clicked.connect(clear_handler)
        row.addWidget(clear_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return group

    def _add_paths(self, list_widget: QListWidget, paths: List[str]) -> None:
        existing = {list_widget.item(i).text() for i in range(list_widget.count())}
        for path in paths:
            norm = str(Path(path))
            if norm not in existing:
                list_widget.addItem(QListWidgetItem(norm))
                existing.add(norm)

    def _remove_selected(self, list_widget: QListWidget) -> None:
        for item in list_widget.selectedItems():
            list_widget.takeItem(list_widget.row(item))

    def _paths_from(self, list_widget: QListWidget) -> List[str]:
        return [list_widget.item(i).text() for i in range(list_widget.count())]

    def add_volumes(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Select MRI volumes',
            '',
            'Images (*.nii *.nii.gz *.nrrd);;All files (*)',
        )
        self._add_paths(self.volume_list, paths)

    def clear_volumes(self) -> None:
        self.volume_list.clear()

    def add_segmentations(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Select segmentations',
            '',
            'Images (*.nii *.nii.gz *.nrrd);;All files (*)',
        )
        self._add_paths(self.segmentation_list, paths)

    def clear_segmentations(self) -> None:
        self.segmentation_list.clear()

    def select_roi(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Select ROI mask',
            '',
            'Images (*.nii *.nii.gz *.nrrd);;All files (*)',
        )
        if path:
            self.roi_edit.setText(str(Path(path)))

    def validate_and_accept(self) -> None:
        if self.tabs.currentWidget() is self.batch_tab:
            if self.batch_tab.discovery_result is None:
                QMessageBox.warning(self, 'Batch Discovery', 'Please scan a directory or load a discovery JSON first.')
                return
            if not (self.batch_tab.discovery_result.launch_selection.selected_suffix or '').strip():
                QMessageBox.warning(self, 'Batch Discovery', 'Please choose a volume type before beginning segmenting.')
                return
            self._launch_mode = 'batch'
        else:
            self._launch_mode = 'manual'
        self.accept()

    def get_config(self) -> StartupConfig:
        if self._launch_mode == 'batch':
            return StartupConfig(
                volume_paths=[],
                segmentation_paths=[],
                segmentation_suffixes=[],
                suffix_to_replace=None,
                roi_mask_path=None,
                batch_discovery_result=self.batch_tab.discovery_result,
                batch_launch_selection=self.batch_tab.discovery_result.launch_selection if self.batch_tab.discovery_result else None,
            )

        suffixes_raw = self.suffixes_edit.text().strip()
        suffixes = suffixes_raw.split() if suffixes_raw else []
        replace_text = self.replace_edit.text().strip() or None
        roi_path = self.roi_edit.text().strip() or None
        return StartupConfig(
            volume_paths=self._paths_from(self.volume_list),
            segmentation_paths=self._paths_from(self.segmentation_list),
            segmentation_suffixes=suffixes,
            suffix_to_replace=replace_text,
            roi_mask_path=roi_path,
            batch_discovery_result=None,
            batch_launch_selection=None,
        )
