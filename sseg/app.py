from __future__ import annotations

import sys
from typing import Optional, Sequence

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

    viewer.show()
    return app.exec_()
