from __future__ import annotations

import argparse
import sys

from PyQt5.QtWidgets import QApplication

from .app import create_viewer_from_config, run_gui
from .startup import StartupConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='sseg', description='3D MRI segmentation viewer')
    sub = parser.add_subparsers(dest='command')

    gui = sub.add_parser('gui', help='Launch the graphical viewer with the startup workflow')
    gui.add_argument('--load-volumes', nargs='+', help='Preload one or more MRI volumes')
    gui.add_argument('--load-segmentations', nargs='+', help='Preload segmentation volumes')
    gui.add_argument('--new-segmentation-suffixes', nargs='+', help='Default suffixes for new segmentations')
    gui.add_argument('--suffix-to-replace', help='Text to replace in first loaded volume filename')
    gui.add_argument('--roi-mask', help='Optional ROI mask path')
    gui.add_argument('--skip-startup', action='store_true', help='Launch the viewer directly when --load-volumes is provided')

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, 'gui'):
        if getattr(args, 'skip_startup', False):
            if not args.load_volumes:
                parser.error('--skip-startup requires --load-volumes')
            app = QApplication.instance() or QApplication(sys.argv)
            config = StartupConfig(
                volume_paths=args.load_volumes,
                segmentation_paths=args.load_segmentations or [],
                segmentation_suffixes=args.new_segmentation_suffixes or [],
                suffix_to_replace=args.suffix_to_replace,
                roi_mask_path=args.roi_mask,
            )
            viewer = create_viewer_from_config(config)
            viewer.show()
            return app.exec_()
        return run_gui(startup=True, argv=argv)

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
