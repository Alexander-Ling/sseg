"""Combo box for picking loaded volumes/segmentations by file name.

Items keep their full name as data but display a shortened label: standard image
extensions are dropped, and when several files are loaded any trailing suffix
they all share (e.g. "_brain-norm") is hidden too. Hovering an item whose label
is shortened or elided shows the full name almost immediately.
"""

import os
import re

from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QToolTip

FILE_EXTENSION_RE = re.compile(r'(\.seg\.nrrd|\.nii\.gz|\.nii|\.nrrd)$', re.IGNORECASE)
SUFFIX_SEPARATORS = "_-. "

NAME_ROLE = Qt.UserRole
SPECIAL_ROLE = Qt.UserRole + 1


def strip_file_extension(name):
    return FILE_EXTENSION_RE.sub('', name)


def compute_display_names(names):
    """Map each full name to a shortened, still-unique display label."""
    stripped = [strip_file_extension(n) or n for n in names]
    if len(stripped) > 1:
        common = os.path.commonprefix([s[::-1] for s in stripped])[::-1]
        # Only hide whole name parts: the hidden suffix must start at a separator,
        # and every name has to keep at least one character.
        shortest = min(len(s) for s in stripped)
        while common and (common[0] not in SUFFIX_SEPARATORS or len(common) >= shortest):
            common = common[1:]
        if common:
            stripped = [s[:-len(common)] for s in stripped]
    # Fall back to full names where shortening made two labels identical (e.g. "a.nii" and "a.nrrd")
    counts = {}
    for s in stripped:
        counts[s] = counts.get(s, 0) + 1
    return {n: (s if counts[s] == 1 else n) for n, s in zip(names, stripped)}


class NameComboBox(QComboBox):
    # Emitted with the item's full name (not its shortened label) when the user picks an item
    nameActivated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.activated[int].connect(self._emit_name_activated)
        self.currentIndexChanged.connect(self._update_own_tooltip)
        self.view().setMouseTracking(True)
        self.view().viewport().installEventFilter(self)

    # Item management, keyed by full name

    def add_name(self, name, special=False):
        self.addItem(name)
        index = self.count() - 1
        self.setItemData(index, name, NAME_ROLE)
        self.setItemData(index, special, SPECIAL_ROLE)
        if not special:
            self.refresh_display_names()

    def remove_name(self, name):
        index = self.find_name(name)
        if index >= 0:
            self.removeItem(index)
            self.refresh_display_names()

    def rename(self, old_name, new_name):
        index = self.find_name(old_name)
        if index >= 0:
            self.setItemData(index, new_name, NAME_ROLE)
            self.refresh_display_names()

    def clear_names(self, keep_special=()):
        self.blockSignals(True)
        self.clear()
        for name in keep_special:
            self.add_name(name, special=True)
        self.blockSignals(False)
        self._update_own_tooltip()

    def find_name(self, name):
        return self.findData(name, NAME_ROLE, Qt.MatchExactly)

    def name_at(self, index):
        return self.itemData(index, NAME_ROLE)

    def current_name(self):
        return self.name_at(self.currentIndex())

    def set_current_name(self, name):
        index = self.find_name(name)
        if index >= 0:
            self.setCurrentIndex(index)

    def refresh_display_names(self):
        indices = [i for i in range(self.count()) if not self.itemData(i, SPECIAL_ROLE)]
        names = [self.name_at(i) for i in indices]
        labels = compute_display_names(names)
        for i, name in zip(indices, names):
            self.setItemText(i, labels[name])
        self._update_own_tooltip()

    # Full-name tooltips

    def _emit_name_activated(self, index):
        name = self.name_at(index)
        if name is not None:
            self.nameActivated.emit(name)

    def _update_own_tooltip(self, *args):
        name = self.current_name()
        self.setToolTip(name if name and name != self.currentText() else "")

    def eventFilter(self, obj, event):
        # Show the full name right away when hovering a shortened or elided popup item,
        # instead of waiting for the default tooltip delay.
        if obj is self.view().viewport():
            if event.type() == QEvent.MouseMove:
                view = self.view()
                index = view.indexAt(event.pos())
                name = index.data(NAME_ROLE) if index.isValid() else None
                if name:
                    label = index.data(Qt.DisplayRole) or ""
                    rect = view.visualRect(index)
                    elided = view.fontMetrics().horizontalAdvance(label) > rect.width() - 8
                    if label != name or elided:
                        QToolTip.showText(event.globalPos(), name, view.viewport(), rect)
                    else:
                        QToolTip.hideText()
                else:
                    QToolTip.hideText()
            elif event.type() == QEvent.ToolTip:
                return True  # tooltips for popup items are handled above
            elif event.type() == QEvent.Leave:
                QToolTip.hideText()
        return super().eventFilter(obj, event)
