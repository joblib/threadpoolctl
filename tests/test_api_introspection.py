"""
Tests for get/set number of threads API introspection.
"""

import sys
from threading import local as threadlocal

import pytest

from threadpoolctl import (
    _ThreadLimitScope,
    _determine_thread_limit_scope,
    ThreadpoolController,
)

# Make sure we have some BLAS libraries loaded:
from . import utils as _


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


def test_determine_thread_limit_scope_thread_local() -> None:
    """
    Check ``_determine_thread_limit_scope()`` can correctly diagnose a trivial
    thread-local implementation.
    """
    api = FakeThreadLocalAPI()
    assert (
        _determine_thread_limit_scope(api.get, api.set)
        == _ThreadLimitScope.CURRENT_THREAD
    )


@pytest.mark.parametrize("default", [1, 17])
def test_determine_thread_limit_scope_processwide(default: int) -> None:
    """
    Check ``_determine_thread_limit_scope()`` can correctly diagnose a trivial
    process-wide implementation.
    """
    api = FakeProcesswideAPI(default)
    assert _determine_thread_limit_scope(api.get, api.set) == _ThreadLimitScope.PROCESS


@pytest.mark.skipif(
    sys.platform != "linux", reason="We only hardcoded Linux-specific behavior"
)
@pytest.mark.parametrize(
    ["select_filter", "expected_thread_limit_scope"],
    [
        (
            {"internal_api": "openblas", "threading_layer": "pthreads"},
            _ThreadLimitScope.PROCESS,
        ),
        (
            {"user_api": "openmp"},
            _ThreadLimitScope.CURRENT_THREAD,
        ),
    ],
)
def test_api_scope(
    select_filter: dict[str, str], expected_thread_limit_scope: str
) -> None:
    """
    Check ``_determine_thread_limit_scope()`` against libraries with known
    properties, to make sure it detects them correctly.  The test is intended
    to be of the function's behavior, not of the libraries.
    """
    controller = ThreadpoolController().select(**select_filter)
    if not controller.lib_controllers:
        pytest.skip(f"{select_filter} controller not found")

    for lib in controller.lib_controllers:
        assert (
            _determine_thread_limit_scope(lib.get_num_threads, lib.set_num_threads)
            == expected_thread_limit_scope
        )
