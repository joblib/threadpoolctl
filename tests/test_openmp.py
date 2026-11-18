"""Introspection of OpenMP runtime settings."""

import ctypes
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from threadpoolctl import (
    _CDLL_CACHE,
    OpenMPController,
    ThreadpoolController,
    threadpool_info,
)

# Import the OpenMP helpers, if available, to load the runtimes used by CI.
from . import utils as _  # noqa: F401


@pytest.mark.parametrize("debugging_info", [False, True])
def test_blocktime_missing_symbol(monkeypatch, debugging_info):
    dynlib = SimpleNamespace(
        omp_get_max_threads=lambda: 2,
        omp_set_num_threads=lambda num_threads: None,
    )
    monkeypatch.setattr(_CDLL_CACHE, "get_cdll", lambda filepath: dynlib)
    controller = OpenMPController(filepath="libgomp.so", prefix="libgomp")

    assert controller.get_blocktime() is None
    assert controller.blocktime_ms is None
    info = controller.info(debugging_info=debugging_info)
    assert info["blocktime_ms"] is None
    assert ("thread_limit_scope" in info) == debugging_info


@pytest.fixture
def kmp_controller():
    controllers = ThreadpoolController().select(user_api="openmp").lib_controllers
    for controller in controllers:
        if hasattr(controller.dynlib, "kmp_get_blocktime") and hasattr(
            controller.dynlib, "kmp_set_blocktime"
        ):
            return controller
    pytest.skip("Requires an OpenMP runtime with the kmp blocktime API")


def test_blocktime_actualized(kmp_controller):
    """A reused controller reflects changes made through the native API."""
    get_blocktime = kmp_controller.dynlib.kmp_get_blocktime
    get_blocktime.argtypes = []
    get_blocktime.restype = ctypes.c_int
    set_blocktime = kmp_controller.dynlib.kmp_set_blocktime
    set_blocktime.argtypes = [ctypes.c_int]
    set_blocktime.restype = None
    controller = ThreadpoolController().select(filepath=kmp_controller.filepath)
    original = get_blocktime()
    try:
        for value in (0, 17):
            set_blocktime(value)
            assert get_blocktime() == value
            assert kmp_controller.get_blocktime() == value
            assert kmp_controller.blocktime_ms == value
            assert controller.info()[0]["blocktime_ms"] == value
            assert controller.select(blocktime_ms=value).info() == controller.info()
            info = next(
                lib
                for lib in threadpool_info()
                if lib["filepath"] == kmp_controller.filepath
            )
            assert info["blocktime_ms"] == value
    finally:
        set_blocktime(original)


def test_blocktime_current_thread(kmp_controller):
    """Reading a shared controller reports the calling thread's blocktime."""
    get_blocktime = kmp_controller.dynlib.kmp_get_blocktime
    get_blocktime.argtypes = []
    get_blocktime.restype = ctypes.c_int
    set_blocktime = kmp_controller.dynlib.kmp_set_blocktime
    set_blocktime.argtypes = [ctypes.c_int]
    set_blocktime.restype = None
    original = get_blocktime()

    def worker():
        worker_original = get_blocktime()
        try:
            set_blocktime(23)
            return kmp_controller.info()["blocktime_ms"], kmp_controller.blocktime_ms
        finally:
            set_blocktime(worker_original)

    try:
        set_blocktime(7)
        assert kmp_controller.info()["blocktime_ms"] == 7
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(worker).result() == (23, 23)
        assert get_blocktime() == 7
        assert kmp_controller.info()["blocktime_ms"] == 7
    finally:
        set_blocktime(original)


@pytest.mark.parametrize("setting", [None, "37", "37ms", "35001us", "35 us"])
def test_blocktime_ms_matches_verbose_runtime_in_subprocess(kmp_controller, setting):
    """Compare introspection with the effective setting printed by the runtime."""
    env = os.environ.copy()
    env["OMP_DISPLAY_ENV"] = "VERBOSE"
    env["KMP_SETTINGS"] = "1"
    env.pop("KMP_BLOCKTIME", None)
    if setting is not None:
        env["KMP_BLOCKTIME"] = setting
    code = """
import ctypes
import json
import os
import sys
from threadpoolctl import threadpool_info

runtime = ctypes.CDLL(sys.argv[1])
runtime.omp_get_max_threads()
initial = threadpool_info()
# The native unit stays fixed when the environment changes after initialization.
os.environ["KMP_BLOCKTIME"] = "1us" if os.environ.get("KMP_BLOCKTIME", "").endswith("ms") else "1ms"
runtime.kmp_set_blocktime(17)
print("THREADPOOLCTL_INFO=" + json.dumps({"initial": initial, "updated": threadpool_info()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, kmp_controller.filepath],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    # Effective settings follow any user settings in KMP_SETTINGS output.
    settings = re.findall(r"KMP_BLOCKTIME\s*=\s*['\"]?(\d+)\s*(us|ms)?", output)
    assert settings, output
    value, unit = settings[-1]
    if setting is not None and setting.endswith("us") and unit != "us":
        pytest.skip("This runtime does not support microsecond KMP_BLOCKTIME settings")
    expected_ms = int(value) / 1000 if unit == "us" else int(value)
    match = re.search(r"^THREADPOOLCTL_INFO=(.*)$", completed.stdout, re.MULTILINE)
    assert match, output
    libraries = json.loads(match.group(1))
    info = next(
        lib
        for lib in libraries["initial"]
        if lib["filepath"] == kmp_controller.filepath
    )
    assert info["blocktime_ms"] == pytest.approx(expected_ms)
    updated = next(
        lib
        for lib in libraries["updated"]
        if lib["filepath"] == kmp_controller.filepath
    )
    assert updated["blocktime_ms"] == pytest.approx(0.017 if unit == "us" else 17)
