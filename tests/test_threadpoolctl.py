import re
import ctypes
import pytest


from threadpoolctl import threadpool_limits, threadpool_info
from threadpoolctl import _ALL_PREFIXES, _ALL_USER_APIS

from .utils import with_check_openmp_num_threads, libopenblas_paths


def should_skip_module(module):
    # Possible bug in getting maximum number of threads with OpenBLAS < 0.2.16
    # and OpenBLAS does not expose its version before 0.3.4.
    return module['internal_api'] == "openblas" and module['version'] is None


@pytest.mark.parametrize("prefix", _ALL_PREFIXES)
def test_threadpool_limits_by_prefix(openblas_present, mkl_present, prefix):
    original_infos = threadpool_info()
    original_num_threads = {info["filepath"]: info['num_threads']
                            for info in original_infos}
    found_mkl = any([True for info in original_infos
                     if info["prefix"] in ('mkl_rt', 'libmkl_rt')])
    prefix_found = len([info["prefix"] for info in original_infos])
    if not prefix_found:
        if "mkl_rt" in prefix and mkl_present and not found_mkl:
            raise RuntimeError("Could not load the MKL prefix")
        elif prefix == "libopenblas" and openblas_present:
            raise RuntimeError("Could not load the OpenBLAS prefix")
        else:
            pytest.skip("Need {} support".format(prefix))

    with threadpool_limits(limits={prefix: 1}):
        for module in threadpool_info():
            if should_skip_module(module):
                continue
            num_threads, filepath = module["num_threads"], module["filepath"]
            if module["prefix"] == prefix:
                assert num_threads == 1
            else:
                assert num_threads == original_num_threads[filepath]

    with threadpool_limits(limits={prefix: 3}):
        for module in threadpool_info():
            if should_skip_module(module):
                continue
            num_threads, filepath = module["num_threads"], module["filepath"]
            if module["prefix"] == prefix:
                expected_num_threads = (3, original_num_threads[filepath])
                assert num_threads in expected_num_threads
            else:
                assert num_threads == original_num_threads[filepath]

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
    original_num_threads = {info["filepath"]: info['num_threads']
                            for info in original_infos}

    with threadpool_limits(limits=1, user_api=user_api):
        for module in threadpool_info():
            if should_skip_module(module):
                continue
            num_threads, filepath = module["num_threads"], module["filepath"]
            if module["user_api"] in user_apis:
                assert num_threads == 1
            else:
                assert num_threads == original_num_threads[filepath]

    with threadpool_limits(limits=3, user_api=user_api):
        for module in threadpool_info():
            if should_skip_module(module):
                continue
            num_threads, filepath = module["num_threads"], module["filepath"]
            if module["user_api"] in user_apis:
                expected_num_threads = (3, original_num_threads[filepath])
                assert num_threads in expected_num_threads
            else:
                assert num_threads == original_num_threads[filepath]

    assert threadpool_info() == original_infos


def test_threadpool_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments
    match = re.escape("user_api must be either in {} or None."
                      .format(_ALL_USER_APIS))
    with pytest.raises(ValueError, match=match):
        threadpool_limits(limits=1, user_api="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int, a dict or None"):
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
