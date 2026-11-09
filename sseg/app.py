from __future__ import annotations

import sys
from typing import Optional, Sequence

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from .startup import StartupDialog, StartupConfig
from .viewer import MRIViewer


def create_viewer_from_config(config: StartupConfig) -> MRIViewer:
    viewer = MRIViewer(
        config.volume_paths,
        config.segmentation_paths or None,
        config.segmentation_suffixes or None,
        [config.suffix_to_replace] if config.suffix_to_replace else None,
        [config.roi_mask_path] if config.roi_mask_path else None,
    )
    viewer.batch_discovery_result = config.batch_discovery_result
    viewer.batch_launch_selection = config.batch_launch_selection
    if config.batch_discovery_result is not None:
        viewer.initialize_batch_navigation()
    return viewer


def show_viewer(viewer: MRIViewer, dialog: Optional[StartupDialog] = None) -> None:
    """Show a viewer whose "Return to Exam Discovery" action reopens the startup dialog.

    The same dialog instance is reused so the scanned exams and batch selections are kept.
    """
    state = {'dialog': dialog}

    def return_to_discovery() -> None:
        app = QApplication.instance()
        # The viewer is the only open window; keep the app alive while it is swapped for the dialog.
        quit_on_last = app.quitOnLastWindowClosed()
        app.setQuitOnLastWindowClosed(False)
        try:
            viewer.dispose()
            startup = state['dialog'] or StartupDialog()
            if startup.batch_tab.discovery_result is not None:
                # Re-analyze so segmentations saved during the session show up in the table.
                startup.batch_tab.set_discovery_result(startup.batch_tab.discovery_result)
            if startup.exec_() != startup.Accepted:
                app.quit()
                return
            show_viewer(create_viewer_from_config(startup.get_config()), startup)
        finally:
            app.setQuitOnLastWindowClosed(quit_on_last)

    # Deferred so the viewer's action handler has returned before the viewer is torn down.
    viewer.return_to_discovery_requested.connect(lambda: QTimer.singleShot(0, return_to_discovery))
    viewer.show()


def run_gui(startup: bool = True, argv: Optional[Sequence[str]] = None) -> int:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else sys.argv)

    if startup:
        dialog = StartupDialog()
        if dialog.exec_() != dialog.Accepted:
            return 0
        config = dialog.get_config()
        viewer = create_viewer_from_config(config)
    else:
        raise ValueError('run_gui(startup=False) requires direct viewer construction by the caller.')

    show_viewer(viewer, dialog)
    return app.exec_()
