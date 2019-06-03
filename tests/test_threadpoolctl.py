import re
import ctypes
import pytest


from threadpoolctl import threadpool_limits, threadpool_info
from threadpoolctl import _ALL_PREFIXES, _ALL_USER_APIS

from .utils import with_check_openmp_num_threads
from .utils import libopenblas_paths
from .utils import scipy


def is_old_openblas(module):
    # Possible bug in getting maximum number of threads with OpenBLAS < 0.2.16
    # and OpenBLAS does not expose its version before 0.3.4.
    return module['internal_api'] == "openblas" and module['version'] is None


def effective_num_threads(nthreads, max_threads):
    if nthreads is None or nthreads > max_threads:
        return max_threads
    return nthreads


@pytest.mark.parametrize("prefix", _ALL_PREFIXES)
def test_threadpool_limits_by_prefix(openblas_present, mkl_present, prefix):
    original_infos = threadpool_info()
    mkl_found = any([True for info in original_infos
                     if info["prefix"] in ('mkl_rt', 'libmkl_rt')])
    prefix_found = len([info["prefix"] for info in original_infos
                        if info["prefix"] == prefix])
    if not prefix_found:
        if "mkl_rt" in prefix and mkl_present and not mkl_found:
            raise RuntimeError("Could not load the MKL prefix")
        elif prefix == "libopenblas" and openblas_present:
            raise RuntimeError("Could not load the OpenBLAS prefix")
        else:
            pytest.skip("{} runtime missing".format(prefix))

    with threadpool_limits(limits={prefix: 1}):
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            if module["prefix"] == prefix:
                assert module["num_threads"] == 1

    with threadpool_limits(limits={prefix: 3}):
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            if module["prefix"] == prefix:
                assert module["num_threads"] <= 3

    assert threadpool_info() == original_infos


@pytest.mark.parametrize("user_api", (None, "blas", "openmp"))
def test_set_threadpool_limits_by_api(user_api):
    # Check that the number of threads used by the multithreaded libraries can
    # be modified dynamically.
    if user_api is None:
        user_apis = ("blas", "openmp")
    else:
        user_apis = (user_api,)

    original_infos = threadpool_info()

    with threadpool_limits(limits=1, user_api=user_api):
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            if module["user_api"] in user_apis:
                assert module["num_threads"] == 1

    with threadpool_limits(limits=3, user_api=user_api):
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            if module["user_api"] in user_apis:
                assert module["num_threads"] <= 3

    assert threadpool_info() == original_infos


def test_threadpool_limits_function_with_side_effect():
    # Check that threadpool_limits can be used as a function with
    # side effects instead of a context manager.
    original_infos = threadpool_info()

    threadpool_limits(limits=1)
    try:
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            assert module["num_threads"] == 1
    finally:
        # Restore the original limits so that this test does not have any
        # side-effect.
        threadpool_limits(limits=original_infos)

    assert threadpool_info() == original_infos


def test_set_threadpool_limits_no_limit():
    # Check that limits=None does nothing.
    original_infos = threadpool_info()
    with threadpool_limits(limits=None):
        assert threadpool_info() == original_infos

    assert threadpool_info() == original_infos


def test_threadpool_limits_manual_unregister():
    # Check that threadpool_limits can be used as an object with that hold
    # the original state of the threadpools that can be restored thanks to the
    # dedicated unregister method
    original_infos = threadpool_info()

    limits = threadpool_limits(limits=1)
    try:
        for module in threadpool_info():
            if is_old_openblas(module):
                continue
            assert module["num_threads"] == 1
    finally:
        # Restore the original limits so that this test does not have any
        # side-effect.
        limits.unregister()

    assert threadpool_info() == original_infos


def test_threadpool_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments
    match = re.escape("user_api must be either in {} or None."
                      .format(_ALL_USER_APIS))
    with pytest.raises(ValueError, match=match):
        threadpool_limits(limits=1, user_api="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a list or a dict"):
        threadpool_limits(limits=(1, 2, 3))


@with_check_openmp_num_threads
@pytest.mark.parametrize('num_threads', [1, 2, 4])
def test_openmp_limit_num_threads(num_threads):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager
    from ._openmp_test_helper import check_openmp_num_threads

    old_num_threads = check_openmp_num_threads(100)

    with threadpool_limits(limits=num_threads):
        assert check_openmp_num_threads(100) in (num_threads, old_num_threads)
    assert check_openmp_num_threads(100) == old_num_threads


@with_check_openmp_num_threads
@pytest.mark.parametrize('nthreads_outer', [None, 1, 2, 4])
def test_openmp_nesting(nthreads_outer):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager
    from ._openmp_test_helper import check_nested_openmp_loops
    from ._openmp_test_helper import get_inner_compiler
    from ._openmp_test_helper import get_outer_compiler

    inner_cc = get_inner_compiler()
    outer_cc = get_outer_compiler()

    outer_num_threads, inner_num_threads = check_nested_openmp_loops(10)

    original_infos = threadpool_info()
    openmp_infos = [info for info in original_infos
                    if info["user_api"] == "openmp"]

    if "gcc" in (inner_cc, outer_cc):
        assert "libgomp" in [info["prefix"] for info in openmp_infos]

    if "clang" in (inner_cc, outer_cc):
        assert "libomp" in [info["prefix"] for info in openmp_infos]

    if inner_cc == outer_cc:
        # The openmp runtime should be shared by default, meaning that
        # the inner loop should automatically be run serially by the OpenMP
        # runtime.
        assert inner_num_threads == 1
    else:
        # There should be at least 2 OpenMP runtime detected.
        assert len(openmp_infos) >= 2

    with threadpool_limits(limits=1) as threadpoolctx:
        max_threads = threadpoolctx.get_original_num_threads('openmp')
        nthreads = effective_num_threads(nthreads_outer, max_threads)

        outer_num_threads, inner_num_threads = \
            check_nested_openmp_loops(10, nthreads)

    # The state of the original state of all threadpools should have been
    # restored.
    assert threadpool_info() == original_infos

    # The number of threads available in the outer loop should not have been
    # decreased:
    assert outer_num_threads == nthreads

    # The number of threads available in the inner loop should have been
    # set to 1 so avoid oversubscription and preserve performance:
    if inner_cc != outer_cc:
        if inner_num_threads != 1:
            # XXX: this does not always work when nesting independent openmp
            # implementations. See: https://github.com/jeremiedbb/Nested_OpenMP
            pytest.xfail("Inner OpenMP num threads was %d instead of 1"
                         % inner_num_threads)
    assert inner_num_threads == 1


def test_shipped_openblas():
    all_openblases = [ctypes.CDLL(path) for path in libopenblas_paths]
    original_num_threads = [blas.openblas_get_num_threads()
                            for blas in all_openblases]

    with threadpool_limits(1):
        for openblas in all_openblases:
            assert openblas.openblas_get_num_threads() == 1

    assert original_num_threads == [openblas.openblas_get_num_threads()
                                    for openblas in all_openblases]


@pytest.mark.skipif(len(libopenblas_paths) < 2,
                    reason="need at least 2 shipped openblas library")
def test_multiple_shipped_openblas():
    # This redundant test is meant to make it easier to see if the system
    # has 2 or more active openblas runtimes available just be reading the
    # pytest report (whether or not this test has been skipped).
    test_shipped_openblas()


@pytest.mark.skipif(scipy is None, reason="requires scipy")
@pytest.mark.parametrize('nthreads_outer', [None, 1, 2, 4])
def test_nested_prange_blas(nthreads_outer):
    import numpy as np

    blas_info = [module for module in threadpool_info()
                 if module["user_api"] == "blas"]
    for module in threadpool_info():
        if is_old_openblas(module):
            # OpenBLAS 0.3.3 and older are known to cause an unrecoverable
            # deadlock at process shutdown time (after pytest has exited).
            pytest.skip("Old OpenBLAS: skipping test to avoid deadlock")

    from ._openmp_test_helper import check_nested_prange_blas
    A = np.ones((1000, 10))
    B = np.ones((100, 10))

    with threadpool_limits(limits=1) as threadpoolctx:
        max_threads = threadpoolctx.get_original_num_threads('openmp')
        nthreads = effective_num_threads(nthreads_outer, max_threads)

        result = check_nested_prange_blas(A, B, nthreads)
        C, prange_num_threads, threadpool_infos = result

    assert np.allclose(C, np.dot(A, B.T))
    assert prange_num_threads == nthreads

    nested_blas_info = [module for module in threadpool_infos
                        if module["user_api"] == "blas"]

    assert len(nested_blas_info) == len(blas_info)
    for module in nested_blas_info:
        assert module['num_threads'] == 1
