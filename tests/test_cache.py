"""Tests for ``threadpoolctl``'s internal caching layer."""

from ctypes import CDLL

import pytest

from threadpoolctl import ThreadpoolController


def test_cdlls_are_cached():
    """
    When a ``LibController`` is created, it reuses the same ``CDLL``.
    """
    pytest.importorskip("numpy")

    controller = ThreadpoolController()
    if not controller.lib_controllers:
        pytest.skip("No libraries loaded")

    cached_cdll = controller.lib_controllers[0].dynlib
    assert isinstance(cached_cdll, CDLL)

    controller2 = ThreadpoolController()
    assert cached_cdll is controller2.lib_controllers[0].dynlib


def test_cache_methods_on_dynlib():
    """
    ``_CDLLCache.cache_method_on_dynlib()`` caches the result, tied to the
    underlying ``CDLL`` as an invalidation key.
    """
    pytest.importorskip("numpy")

    # We assume all BLAS libs have ``get_version()`` wrapped with
    # ``cache_method_on_dynlib()``, which is currently the case.
    controller = ThreadpoolController()
    libs = controller.select(user_api="blas").lib_controllers
    if not libs:
        pytest.skip("No libraries loaded")

    assert libs[0].get_version() is libs[0].get_version()

    # Access underlying, uncached get_version():
    assert libs[0].get_version.__wrapped__(libs[0]) is not libs[0].get_version()
