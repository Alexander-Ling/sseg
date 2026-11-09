from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
)


class GlobalHotkeyFilter(QObject):
    """Global event filter for temporary segmentation hiding via the A key."""

    def __init__(self, mri_viewer):
        super().__init__()
        self.mri_viewer = mri_viewer
        self.hiding_active = False

    def _hotkey_allowed(self):
        """Only trigger the hotkey when the viewer window is active and no text field has focus."""
        if QApplication.activeWindow() is not self.mri_viewer:
            return False

        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            return False
        if isinstance(focus, QComboBox) and focus.isEditable():
            return False
        return True

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_A:
            if not self.hiding_active and not self._hotkey_allowed():
                return False
            if not event.isAutoRepeat() and not self.hiding_active:
                self.hiding_active = True
                self.mri_viewer.backup_invisible_levels = set(self.mri_viewer.invisible_levels)
                self.mri_viewer.invisible_levels = set(self.mri_viewer.unique_levels)
                self.mri_viewer.display_all_slices()
            return True

        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_A:
            if not self.hiding_active:
                return False
            if not event.isAutoRepeat():
                self.hiding_active = False
                self.mri_viewer.invisible_levels = set(self.mri_viewer.backup_invisible_levels)
                self.mri_viewer.backup_invisible_levels.clear()
                self.mri_viewer.display_all_slices()
            return True

        return False
