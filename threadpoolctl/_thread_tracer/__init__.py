"""Private experimental thread spawn tracer (not part of the public API)."""

import sys

from threadpoolctl._thread_tracer._types import ThreadSpawnStats, ThreadTracerError
from threadpoolctl._thread_tracer._windows_etw import WindowsThreadSpawnTracer


def windows_tracer_available():
    """Return whether the Windows ETW tracer can run on this platform."""
    return sys.platform == "win32"


def windows_etw_admin_available():
    """Return whether the current process can start kernel ETW sessions."""
    if not windows_tracer_available():
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        return False


__all__ = [
    "ThreadSpawnStats",
    "ThreadTracerError",
    "WindowsThreadSpawnTracer",
    "windows_etw_admin_available",
    "windows_tracer_available",
]
