"""Private experimental thread spawn tracer (not part of the public API)."""

import sys

from threadpoolctl._thread_tracer._types import ThreadSpawnStats, ThreadTracerError
from threadpoolctl._thread_tracer._windows_etw import WindowsThreadSpawnTracer


def windows_tracer_available():
    """Return whether the Windows ETW tracer can run on this platform."""
    return sys.platform == "win32"


__all__ = [
    "ThreadSpawnStats",
    "ThreadTracerError",
    "WindowsThreadSpawnTracer",
    "windows_tracer_available",
]
