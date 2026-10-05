"""Tests for ``threadpoolctl``'s internal caching layer."""

from ctypes import CDLL

import pytest

from threadpoolctl import ThreadpoolController, _CDLL_CACHE


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


def test_cache_methods_on_dynlib(request):
    """
    ``_CDLLCache.cache_method_on_dynlib()`` caches the result, tied to the
    underlying ``CDLL`` as an invalidation key.
    """
    pytest.importorskip("numpy")

    controller = ThreadpoolController()
    libs = controller.select(user_api="blas").lib_controllers
    if not libs:
        pytest.skip("No libraries loaded")

    @_CDLL_CACHE.cache_method_on_dynlib
    def my_extra_method(self):
        return object()

    # Can't use pytest's monkeypatch since this method doesn't already exist,
    # so add it manually:
    libs[0].__class__.my_extra_method = my_extra_method

    def cleanup():
        del libs[0].__class__.my_extra_method

    request.addfinalizer(cleanup)

    # Accessing underlying, uncached method returns different objects each time:
    initial = libs[0].my_extra_method()
    assert libs[0].my_extra_method.__wrapped__(libs[0]) is not initial
    assert libs[0].my_extra_method.__wrapped__(libs[0]) is not initial

    # Only one call should ever happen when using the cached method, however:
    assert libs[0].my_extra_method() is initial
    assert libs[0].my_extra_method() is initial
