from __future__ import annotations

from PyQt5.QtWidgets import QMenu


class PersistentMenu(QMenu):
    """A checkable menu that stays open while toggling items."""

    def __init__(self, title, parent=None, tear_off=True):
        super().__init__(title, parent)
        self.setTearOffEnabled(tear_off)

    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.pos())
        # Checkable items, and items flagged with property keepOpen (e.g. "Show all"), do not close the menu
        if action and action.isEnabled() and (action.isCheckable() or action.property("keepOpen")):
            action.trigger()
            return
        super().mouseReleaseEvent(event)
