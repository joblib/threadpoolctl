"""Tests for ``threadpoolctl``'s internal caching layer."""

from ctypes import CDLL

import pytest

from threadpoolctl import _CachingCDLL, _CDLLCache, ThreadpoolController


class FakeCDLL:
    """A stand-in for a real ``CDLL``."""

    @property
    def real_attr(self):
        return (object(), 123)


def test_caching_cdll():
    """
    Attributes of a ``_CachingCDLL`` are retrieved from the wrapped ``CDLL``
    only once.
    """
    fake_dll = FakeCDLL()
    # The attribute is created from scratch each time:
    assert fake_dll.real_attr is not fake_dll.real_attr

    # But not when cached!
    python_cache = _CachingCDLL(fake_dll)
    assert python_cache.real_attr is python_cache.real_attr
    assert python_cache.real_attr[1] == 123

    with pytest.raises(AttributeError):
        python_cache.none_such

    assert not hasattr(python_cache, "none_such")


def test_cdlls_are_cached():
    """
    When a ``LibController`` is created, it reuses the same ``CDLL``.
    """
    pytest.importorskip("numpy")

    controller = ThreadpoolController()
    cached_cdll = controller.lib_controllers[0].dynlib
    assert isinstance(cached_cdll, _CachingCDLL)
    assert isinstance(cached_cdll._cdll, CDLL)

    controller2 = ThreadpoolController()
    assert cached_cdll._cdll is controller2.lib_controllers[0].dynlib._cdll


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
    assert libs[0].get_version() is libs[0].get_version()

    # Access underlying, uncached get_version():
    assert libs[0].get_version.__wrapped__(libs[0]) is not libs[0].get_version()
