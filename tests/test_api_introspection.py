"""
Tests for get/set number of threads API introspection.
"""

from threading import local as threadlocal

import pytest

from threadpoolctl import _APIScope, _determine_api_scope, ThreadpoolController


class FakeThreadLocalAPI(threadlocal):
    """Thread-local num threads setting API."""

    def get(self) -> int:
        return getattr(self, "num_threads", 17)

    def set(self, n: int) -> None:
        self.num_threads = n


class FakeProcesswideAPI:
    """Process-wide num threads setting API."""

    def __init__(self, num_threads: int):
        self.num_threads = num_threads

    def get(self) -> int:
        return self.num_threads

    def set(self, n: int) -> None:
        self.num_threads = n


def test_determine_api_scope_thread_local():
    """
    Check ``_determine_api_scope()`` can correctly diagnose a trivial
    thread-local implementation.
    """
    api = FakeThreadLocalAPI()
    assert _determine_api_scope(api.get, api.set) == _APIScope.CURRENT_THREAD


@pytest.mark.parametrize("default", [1, 17])
def test_determine_api_scope_processiwde(default: int):
    """
    Check ``_determine_api_scope()`` can correctly diagnose a trivial
    process-wide implementation.
    """
    api = FakeProcesswideAPI(default)
    assert _determine_api_scope(api.get, api.set) == _APIScope.PROCESS


@pytest.mark.parametrize(
    ["select_filter", "expected_api_scope"],
    [
        (
            {"internal_api": "openblas", "threading_layer": "pthreads"},
            _APIScope.PROCESS,
        ),
        ({"user_api": "openmp", "prefix": "libgomp"}, _APIScope.CURRENT_THREAD),
        ({"user_api": "openmp", "prefix": "libomp"}, _APIScope.CURRENT_THREAD),
    ],
)
def test_api_scope(select_filter: dict[str, str], expected_api_scope: str) -> None:
    """
    Check ``determine_api_scope()`` against known values, to make sure it
    detects them correctly.
    """
    controller = ThreadpoolController().select(**select_filter)
    if not controller.lib_controllers:
        pytest.skip(f"{select_filter} controller not found")

    for lib in controller.lib_controllers:
        assert (
            _determine_api_scope(lib.get_num_threads, lib.set_num_threads)
            == expected_api_scope
        )
