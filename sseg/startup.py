from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QWidget,
    QFormLayout, QGroupBox
)


@dataclass
class StartupConfig:
    volume_paths: List[str]
    segmentation_paths: List[str]
    segmentation_suffixes: List[str]
    suffix_to_replace: Optional[str]
    roi_mask_path: Optional[str]


class StartupDialog(QDialog):
    """Startup workflow for package-based GUI launch.

    This dialog replaces the old requirement that users must launch the viewer
    with ``--load_volumes`` from the command line.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('sseg startup')
        self.resize(760, 560)

        root = QVBoxLayout(self)
        intro = QLabel(
            'Open one or more MRI volumes to start a session, or launch an empty viewer for batch navigation. '            'You can also preload existing segmentations, an ROI mask, and '            'default output naming options.'
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

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        launch = QPushButton('Launch viewer')
        launch.clicked.connect(self.validate_and_accept)
        launch.setDefault(True)
        buttons.addWidget(launch)
        root.addLayout(buttons)

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
        self.accept()

    def get_config(self) -> StartupConfig:
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
        )
