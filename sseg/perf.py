"""Opt-in timing helper. Set SSEG_PERF=1 to print per-call timings for the hot paths."""

import functools
import os
import time

ENABLED = os.environ.get("SSEG_PERF", "") not in ("", "0")


def timed(name):
    """Decorator that logs the wall time of a method when SSEG_PERF is set (no-op otherwise)."""
    def decorator(func):
        if not ENABLED:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                print(f"[perf] {name}: {(time.perf_counter() - start) * 1000:.2f} ms", flush=True)
        return wrapper
    return decorator
