"""Introspection of OpenMP runtime settings."""

import ctypes
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
    assert controller.blocktime is None
    info = controller.info(debugging_info=debugging_info)
    assert info["blocktime"] is None
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
            assert kmp_controller.blocktime == value
            assert controller.info()[0]["blocktime"] == value
            assert controller.select(blocktime=value).info() == controller.info()
            info = next(
                lib
                for lib in threadpool_info()
                if lib["filepath"] == kmp_controller.filepath
            )
            assert info["blocktime"] == value
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
            return kmp_controller.info()["blocktime"], kmp_controller.blocktime
        finally:
            set_blocktime(worker_original)

    try:
        set_blocktime(7)
        assert kmp_controller.info()["blocktime"] == 7
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(worker).result() == (23, 23)
        assert get_blocktime() == 7
        assert kmp_controller.info()["blocktime"] == 7
    finally:
        set_blocktime(original)
