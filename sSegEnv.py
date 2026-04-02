"""Backward-compatible launcher for the packaged sseg application."""
from sseg.cli import main

if __name__ == '__main__':
    raise SystemExit(main(['gui', '--skip-startup', *(__import__('sys').argv[1:])]))
