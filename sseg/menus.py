from __future__ import annotations

from PyQt5.QtWidgets import QMenu


class PersistentMenu(QMenu):
    """A checkable menu that stays open while toggling items."""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setTearOffEnabled(True)

    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.pos())
        if action and action.isCheckable():
            action.trigger()
            return
        super().mouseReleaseEvent(event)
