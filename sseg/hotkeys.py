from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt


class GlobalHotkeyFilter(QObject):
    """Global event filter for temporary segmentation hiding via the A key."""

    def __init__(self, mri_viewer):
        super().__init__()
        self.mri_viewer = mri_viewer

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self.mri_viewer.backup_invisible_levels = set(self.mri_viewer.invisible_levels)
                self.mri_viewer.invisible_levels = set(self.mri_viewer.unique_levels)
                self.mri_viewer.display_all_slices()
            return True

        if event.type() == QEvent.KeyRelease and event.key() == Qt.Key_A:
            if not event.isAutoRepeat():
                self.mri_viewer.invisible_levels = set(self.mri_viewer.backup_invisible_levels)
                self.mri_viewer.backup_invisible_levels.clear()
                self.mri_viewer.display_all_slices()
            return True

        return False
