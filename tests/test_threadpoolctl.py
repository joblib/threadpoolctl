import json
import os
import pytest
import re
import subprocess
import sys

from threadpoolctl import threadpool_limits, threadpool_info
from threadpoolctl import ThreadpoolController
from threadpoolctl import _ALL_PREFIXES, _ALL_USER_APIS

from .utils import cython_extensions_compiled
from .utils import check_nested_prange_blas
from .utils import libopenblas_paths
from .utils import scipy
from .utils import threadpool_info_from_subprocess
from .utils import select


def is_old_openblas(lib_controller):
    # Possible bug in getting maximum number of threads with OpenBLAS < 0.2.16
    # and OpenBLAS does not expose its version before 0.3.4.
    return lib_controller.internal_api == "openblas" and lib_controller.version is None


def skip_if_openblas_openmp():
    """Helper to skip tests with side effects when OpenBLAS has the OpenMP
    threading layer.
    """
    if any(
        lib_controller.internal_api == "openblas"
        and lib_controller.threading_layer == "openmp"
        for lib_controller in ThreadpoolController().lib_controllers
    ):
        pytest.skip(
            "Setting a limit on OpenBLAS when using the OpenMP threading layer also "
            "impact the OpenMP library. They can't be controlled independently."
        )


def effective_num_threads(nthreads, max_threads):
    if nthreads is None or nthreads > max_threads:
        return max_threads
    return nthreads


def test_threadpool_info():
    # Check consistency between threadpool_info and ThreadpoolController
    function_info = threadpool_info()
    object_info = ThreadpoolController().lib_controllers

    for lib_info, lib_controller in zip(function_info, object_info):
        assert lib_info == lib_controller.info()


def test_threadpool_controller_info():
    # Check that all keys expected for the private api are in the dicts
    # returned by the `info` methods
    controller = ThreadpoolController()

    assert threadpool_info() == [
        lib_controller.info() for lib_controller in controller.lib_controllers
    ]
    assert controller.info() == [
        lib_controller.info() for lib_controller in controller.lib_controllers
    ]

    for lib_controller_dict in controller.info():
        assert "user_api" in lib_controller_dict
        assert "internal_api" in lib_controller_dict
        assert "prefix" in lib_controller_dict
        assert "filepath" in lib_controller_dict
        assert "version" in lib_controller_dict
        assert "num_threads" in lib_controller_dict

        if lib_controller_dict["internal_api"] in ("mkl", "blis", "openblas"):
            assert "threading_layer" in lib_controller_dict


def test_controller_info_actualized():
    # Check that the num_threads attribute reflects the actual state of the threadpools
    controller = ThreadpoolController()
    original_info = controller.info()

    with threadpool_limits(limits=1):
        assert all(
            lib_controller.num_threads == 1
            for lib_controller in controller.lib_controllers
        )

    assert controller.info() == original_info


@pytest.mark.parametrize(
    "kwargs",
    [
        {"user_api": "blas"},
        {"prefix": "libgomp"},
        {"internal_api": "openblas", "prefix": "libomp"},
        {"prefix": ["libgomp", "libomp", "libiomp"]},
    ],
)
def test_threadpool_controller_select(kwargs):
    # Check the behior of the select method of ThreadpoolController
    controller = ThreadpoolController().select(**kwargs)
    if not controller:
        pytest.skip(f"Requires at least one of {list(kwargs.values())}.")

    for lib_controller in controller.lib_controllers:
        assert any(
            getattr(lib_controller, key) in (val if isinstance(val, list) else [val])
            for key, val in kwargs.items()
        )


@pytest.mark.parametrize("prefix", _ALL_PREFIXES)
@pytest.mark.parametrize("limit", [1, 3])
def test_threadpool_limits_by_prefix(prefix, limit):
    # Check that the maximum number of threads can be set by prefix
    controller = ThreadpoolController()
    original_info = controller.info()

    controller_matching_prefix = controller.select(prefix=prefix)
    if not controller_matching_prefix:
        pytest.skip(f"Requires {prefix} runtime")

    with threadpool_limits(limits={prefix: limit}):
        for lib_controller in controller_matching_prefix.lib_controllers:
            if is_old_openblas(lib_controller):
                continue
            # threadpool_limits only sets an upper bound on the number of
            # threads.
            assert 0 < lib_controller.num_threads <= limit
    assert ThreadpoolController().info() == original_info


@pytest.mark.parametrize("user_api", (None, "blas", "openmp"))
@pytest.mark.parametrize("limit", [1, 3])
def test_set_threadpool_limits_by_api(user_api, limit):
    # Check that the maximum number of threads can be set by user_api
    controller = ThreadpoolController()
    original_info = controller.info()

    if user_api is None:
        controller_matching_api = controller
    else:
        controller_matching_api = controller.select(user_api=user_api)
    if not controller_matching_api:
        user_apis = _ALL_USER_APIS if user_api is None else [user_api]
        pytest.skip(f"Requires a library which api is in {user_apis}")

    with threadpool_limits(limits=limit, user_api=user_api):
        for lib_controller in controller_matching_api.lib_controllers:
            if is_old_openblas(lib_controller):
                continue
            # threadpool_limits only sets an upper bound on the number of
            # threads.
            assert 0 < lib_controller.num_threads <= limit

    assert ThreadpoolController().info() == original_info


def test_threadpool_limits_function_with_side_effect():
    # Check that threadpool_limits can be used as a function with
    # side effects instead of a context manager.
    original_info = ThreadpoolController().info()

    threadpool_limits(limits=1)
    try:
        for lib_controller in ThreadpoolController().lib_controllers:
            if is_old_openblas(lib_controller):
                continue
            assert lib_controller.num_threads == 1
    finally:
        # Restore the original limits so that this test does not have any
        # side-effect.
        threadpool_limits(limits=original_info)

    assert ThreadpoolController().info() == original_info


def test_set_threadpool_limits_no_limit():
    # Check that limits=None does nothing.
    original_info = ThreadpoolController().info()

    with threadpool_limits(limits=None):
        assert ThreadpoolController().info() == original_info

    assert ThreadpoolController().info() == original_info


def test_threadpool_limits_manual_restore():
    # Check that threadpool_limits can be used as an object which holds the
    # original state of the threadpools and that can be restored thanks to the
    # dedicated restore_original_limits method
    original_info = ThreadpoolController().info()

    limits = threadpool_limits(limits=1)
    try:
        for lib_controller in ThreadpoolController().lib_controllers:
            if is_old_openblas(lib_controller):
                continue
            assert lib_controller.num_threads == 1
    finally:
        # Restore the original limits so that this test does not have any
        # side-effect.
        limits.restore_original_limits()

    assert ThreadpoolController().info() == original_info


def test_threadpool_controller_limit():
    # Check that using the limit method of ThreadpoolController only impact its
    # library controllers.

    # This is not True for OpenBLAS with the OpenMP threading layer.
    skip_if_openblas_openmp()

    blas_controller = ThreadpoolController().select(user_api="blas")
    original_openmp_info = ThreadpoolController().select(user_api="openmp").info()

    with blas_controller.limit(limits=1):
        blas_controller = ThreadpoolController().select(user_api="blas")
        openmp_info = ThreadpoolController().select(user_api="openmp").info()

        assert all(
            lib_controller.num_threads == 1
            for lib_controller in blas_controller.lib_controllers
        )
        # original_blas_controller contains only blas libraries so no opemp library
        # should be impacted.
        assert openmp_info == original_openmp_info


def test_get_params_for_sequential_blas_under_openmp():
    # Test for the behavior of get_params_for_sequential_blas_under_openmp.
    controller = ThreadpoolController()
    original_info = controller.info()

    params = controller._get_params_for_sequential_blas_under_openmp()

    if controller.select(
        internal_api="openblas", threading_layer="openmp"
    ).lib_controllers:
        assert params["limits"] is None
        assert params["user_api"] is None

        with controller.limit(limits="sequential_blas_under_openmp"):
            assert controller.info() == original_info

    else:
        assert params["limits"] == 1
        assert params["user_api"] == "blas"

        with controller.limit(limits="sequential_blas_under_openmp"):
            assert all(
                lib_info["num_threads"] == 1
                for lib_info in controller.info()
                if lib_info["user_api"] == "blas"
            )


def test_nested_limits():
    # Check that exiting the context manager properly restores the original limits even
    # when nested.
    controller = ThreadpoolController()
    original_info = controller.info()

    if any(info["num_threads"] < 2 for info in original_info):
        pytest.skip("Test requires at least 2 CPUs on host machine")

    def check_num_threads(expected_num_threads):
        assert all(
            lib_controller.num_threads == expected_num_threads
            for lib_controller in ThreadpoolController().lib_controllers
        )

    with controller.limit(limits=1):
        check_num_threads(expected_num_threads=1)

        with controller.limit(limits=2):
            check_num_threads(expected_num_threads=2)

        check_num_threads(expected_num_threads=1)

    assert ThreadpoolController().info() == original_info


def test_threadpool_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments
    match = re.escape(f"user_api must be either in {_ALL_USER_APIS} or None.")
    with pytest.raises(ValueError, match=match):
        threadpool_limits(limits=1, user_api="wrong")

    with pytest.raises(
        TypeError, match="limits must either be an int, a list, a dict, or"
    ):
        threadpool_limits(limits=(1, 2, 3))


@pytest.mark.skipif(
    not cython_extensions_compiled, reason="Requires cython extensions to be compiled"
)
@pytest.mark.parametrize("num_threads", [1, 2, 4])
def test_openmp_limit_num_threads(num_threads):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager
    import tests._openmp_test_helper.openmp_helpers_inner as omp_inner

    check_openmp_num_threads = omp_inner.check_openmp_num_threads

    old_num_threads = check_openmp_num_threads(100)

    with threadpool_limits(limits=num_threads):
        assert check_openmp_num_threads(100) in (num_threads, old_num_threads)
    assert check_openmp_num_threads(100) == old_num_threads


@pytest.mark.skipif(
    not cython_extensions_compiled, reason="Requires cython extensions to be compiled"
)
@pytest.mark.parametrize("nthreads_outer", [None, 1, 2, 4])
def test_openmp_nesting(nthreads_outer):
    # checks that OpenMP effectively uses the number of threads requested by
    # the context manager when nested in an outer OpenMP loop.
    import tests._openmp_test_helper.openmp_helpers_outer as omp_outer

    check_nested_openmp_loops = omp_outer.check_nested_openmp_loops

    # Find which OpenMP lib is used at runtime for inner loop
    inner_info = threadpool_info_from_subprocess(
        "tests._openmp_test_helper.openmp_helpers_inner"
    )
    assert len(inner_info) == 1
    inner_omp = inner_info[0]["prefix"]

    # Find which OpenMP lib is used at runtime for outer loop
    outer_info = threadpool_info_from_subprocess(
        "tests._openmp_test_helper.openmp_helpers_outer"
    )
    if len(outer_info) == 1:
        # Only 1 openmp loaded. It has to be this one.
        outer_omp = outer_info[0]["prefix"]
    else:
        # There are 2 openmp, the one from inner and the one from outer.
        assert len(outer_info) == 2
        # We already know the one from inner. It has to be the other one.
        prefixes = {lib_info["prefix"] for lib_info in outer_info}
        outer_omp = prefixes - {inner_omp}

    outer_num_threads, inner_num_threads = check_nested_openmp_loops(10)
    original_info = ThreadpoolController().info()

    if inner_omp == outer_omp:
        # The OpenMP runtime should be shared by default, meaning that the
        # inner loop should automatically be run serially by the OpenMP runtime
        assert inner_num_threads == 1

    with threadpool_limits(limits=1) as threadpoolctx:
        max_threads = threadpoolctx.get_original_num_threads()["openmp"]
        nthreads = effective_num_threads(nthreads_outer, max_threads)

        # Ask outer loop to run on nthreads threads and inner loop run on 1
        # thread
        outer_num_threads, inner_num_threads = check_nested_openmp_loops(10, nthreads)

    # The state of the original state of all threadpools should have been
    # restored.
    assert ThreadpoolController().info() == original_info

    # The number of threads available in the outer loop should not have been
    # decreased:
    assert outer_num_threads == nthreads

    # The number of threads available in the inner loop should have been
    # set to 1 to avoid oversubscription and preserve performance:
    if inner_omp != outer_omp:
        if inner_num_threads != 1:
            # XXX: this does not always work when nesting independent openmp
            # implementations. See: https://github.com/jeremiedbb/Nested_OpenMP
            pytest.xfail(
                f"Inner OpenMP num threads was {inner_num_threads} instead of 1"
            )
    assert inner_num_threads == 1


def test_shipped_openblas():
    # checks that OpenBLAS effectively uses the number of threads requested by
    # the context manager
    original_info = ThreadpoolController().info()
    openblas_controller = ThreadpoolController().select(internal_api="openblas")

    with threadpool_limits(1):
        for lib_controller in openblas_controller.lib_controllers:
            assert lib_controller.num_threads == 1

    assert ThreadpoolController().info() == original_info


@pytest.mark.skipif(
    len(libopenblas_paths) < 2, reason="need at least 2 shipped openblas library"
)
def test_multiple_shipped_openblas():
    # This redundant test is meant to make it easier to see if the system
    # has 2 or more active openblas runtimes available just by reading the
    # pytest report (whether or not this test has been skipped).
    test_shipped_openblas()


@pytest.mark.skipif(
    not cython_extensions_compiled, reason="Requires cython extensions to be compiled"
)
@pytest.mark.skipif(
    check_nested_prange_blas is None,
    reason="Requires nested_prange_blas to be compiled",
)
@pytest.mark.parametrize("nthreads_outer", [None, 1, 2, 4])
def test_nested_prange_blas(nthreads_outer):
    # Check that the BLAS uses the number of threads requested by the context manager
    # when nested in an outer OpenMP loop.
    # Remark: this test also works with sequential BLAS only because we limit the
    # number of threads for the BLAS to 1.
    import numpy as np

    skip_if_openblas_openmp()

    original_info = ThreadpoolController().info()

    blas_controller = ThreadpoolController().select(user_api="blas")
    blis_controller = ThreadpoolController().select(internal_api="blis")

    # skip if the BLAS used by numpy is an old openblas. OpenBLAS 0.3.3 and
    # older are known to cause an unrecoverable deadlock at process shutdown
    # time (after pytest has exited).
    # numpy can be linked to BLIS for CBLAS and OpenBLAS for LAPACK. In that
    # case this test will run BLIS gemm so no need to skip.
    if not blis_controller and any(
        is_old_openblas(lib_controller)
        for lib_controller in blas_controller.lib_controllers
    ):
        pytest.skip("Old OpenBLAS: skipping test to avoid deadlock")

    A = np.ones((1000, 10))
    B = np.ones((100, 10))

    with threadpool_limits(limits=1) as threadpoolctx:
        max_threads = threadpoolctx.get_original_num_threads()["openmp"]
        nthreads = effective_num_threads(nthreads_outer, max_threads)

        result = check_nested_prange_blas(A, B, nthreads)
        C, prange_num_threads, inner_info = result

    assert np.allclose(C, np.dot(A, B.T))
    assert prange_num_threads == nthreads

    nested_blas_info = select(inner_info, user_api="blas")
    assert len(nested_blas_info) == len(blas_controller.lib_controllers)
    assert all(lib_info["num_threads"] == 1 for lib_info in nested_blas_info)

    assert ThreadpoolController().info() == original_info


# the method `get_original_num_threads` raises a UserWarning due to different
# num_threads from libraries with the same `user_api`. It will be raised only
# in the CI job with 2 openblas (py38_pip_openblas_gcc_clang). It is expected
# so we can safely filter it.
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("limit", [1, None])
def test_get_original_num_threads(limit):
    # Tests the method get_original_num_threads of the context manager
    with threadpool_limits(limits=2, user_api="blas") as ctx:
        # set different blas num threads to start with (when multiple openblas)
        if len(ctx._controller.select(user_api="blas")) > 1:
            ctx._controller.lib_controllers[0].set_num_threads(1)

        original_info = ThreadpoolController().info()
        with threadpool_limits(limits=limit, user_api="blas") as threadpoolctx:
            original_num_threads = threadpoolctx.get_original_num_threads()

            assert "openmp" not in original_num_threads

            blas_info = select(original_info, user_api="blas")
            if blas_info:
                expected = min(lib_info["num_threads"] for lib_info in blas_info)
                assert original_num_threads["blas"] == expected
            else:
                assert original_num_threads["blas"] is None

            if len(libopenblas_paths) >= 2:
                with pytest.warns(None, match="Multiple value possible"):
                    threadpoolctx.get_original_num_threads()


def test_mkl_threading_layer():
    # Check that threadpool_info correctly recovers the threading layer used
    # by mkl
    mkl_controller = ThreadpoolController().select(internal_api="mkl")
    expected_layer = os.getenv("MKL_THREADING_LAYER")

    if not (mkl_controller and expected_layer):
        pytest.skip("requires MKL and the environment variable MKL_THREADING_LAYER set")

    actual_layer = mkl_controller.lib_controllers[0].threading_layer
    assert actual_layer == expected_layer.lower()


def test_blis_threading_layer():
    # Check that threadpool_info correctly recovers the threading layer used
    # by blis
    blis_controller = ThreadpoolController().select(internal_api="blis")
    expected_layer = os.getenv("BLIS_ENABLE_THREADING")
    if expected_layer == "no":
        expected_layer = "disabled"

    if not (blis_controller and expected_layer):
        pytest.skip(
            "requires BLIS and the environment variable BLIS_ENABLE_THREADING set"
        )

    actual_layer = blis_controller.lib_controllers[0].threading_layer
    assert actual_layer == expected_layer


@pytest.mark.skipif(
    not cython_extensions_compiled, reason="Requires cython extensions to be compiled"
)
def test_libomp_libiomp_warning(recwarn):
    # Trigger the import of a potentially clang-compiled extension:
    import tests._openmp_test_helper.openmp_helpers_outer  # noqa

    # Trigger the import of numpy to potentially import Intel OpenMP via MKL
    pytest.importorskip("numpy.linalg")

    # Check that a warning is raised when both libomp and libiomp are loaded
    # It should happen in one CI job (pylatest_conda_mkl_clang_gcc).
    controller = ThreadpoolController()
    prefixes = [lib_controller.prefix for lib_controller in controller.lib_controllers]

    if not ("libomp" in prefixes and "libiomp" in prefixes and sys.platform == "linux"):
        pytest.skip("Requires both libomp and libiomp loaded, on Linux")

    assert len(recwarn) == 1
    wm = recwarn[0]
    assert wm.category == RuntimeWarning
    assert "Found Intel" in str(wm.message)
    assert "LLVM" in str(wm.message)
    assert "multiple_openmp.md" in str(wm.message)


def test_command_line_empty_or_system_openmp():
    # When the command line is called without arguments, no library should be
    # detected. The only exception is a system OpenMP library that can be
    # linked to the Python interpreter, for instance via the libb2.so BLAKE2
    # library that can itself be linked to an OpenMP runtime on Gentoo.
    output = subprocess.check_output((sys.executable + " -m threadpoolctl").split())
    results = json.loads(output.decode("utf-8"))
    conda_prefix = os.getenv("CONDA_PREFIX")
    managed_by_conda = conda_prefix and sys.executable.startswith(conda_prefix)
    if not managed_by_conda:  # pragma: no cover
        # When using a Python interpreter that does not come from a conda
        # environment, we should ignore any system OpenMP library.
        results = [r for r in results if r["user_api"] != "openmp"]
    assert results == []


def test_command_line_command_flag():
    pytest.importorskip("numpy")
    output = subprocess.check_output(
        [sys.executable, "-m", "threadpoolctl", "-c", "import numpy"]
    )
    cli_info = json.loads(output.decode("utf-8"))

    this_process_info = threadpool_info()
    for lib_info in cli_info:
        assert lib_info in this_process_info


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="need recent subprocess.run options"
)
def test_command_line_import_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "threadpoolctl",
            "-i",
            "numpy",
            "scipy.linalg",
            "invalid_package",
            "numpy.invalid_sumodule",
        ],
        capture_output=True,
        check=True,
        encoding="utf-8",
    )
    cli_info = json.loads(result.stdout)

    this_process_info = threadpool_info()
    for lib_info in cli_info:
        assert lib_info in this_process_info

    warnings = [w.strip() for w in result.stderr.splitlines()]
    assert "WARNING: could not import invalid_package" in warnings
    assert "WARNING: could not import numpy.invalid_sumodule" in warnings
    if scipy is None:
        assert "WARNING: could not import scipy.linalg" in warnings
    else:
        assert "WARNING: could not import scipy.linalg" not in warnings


def test_architecture():
    expected_openblas_architectures = (
        # XXX: add more as needed by CI or developer laptops
        "armv8",
        "haswell",
        "prescott",  # see: https://github.com/xianyi/OpenBLAS/pull/3485
        "skylakex",
        "sandybridge",
        "vortex",
        "zen",
    )
    expected_blis_architectures = (
        # XXX: add more as needed by CI or developer laptops
        "skx",
        "haswell",
    )
    for lib_info in threadpool_info():
        if lib_info["internal_api"] == "openblas":
            assert lib_info["architecture"].lower() in expected_openblas_architectures
        elif lib_info["internal_api"] == "blis":
            assert lib_info["architecture"].lower() in expected_blis_architectures
        else:
            # Not supported for other libraries
            assert "architecture" not in lib_info


def test_openblas_threading_layer():
    # Check that threadpool_info correctly recovers the threading layer used by openblas
    openblas_controller = ThreadpoolController().select(internal_api="openblas")

    if not (openblas_controller):
        pytest.skip("requires OpenBLAS.")

    expected_openblas_threading_layers = ("openmp", "pthreads", "disabled")

    threading_layer = openblas_controller.lib_controllers[0].threading_layer

    if threading_layer == "unknown":
        # If we never recover an acceptable value for the threading layer, it will be
        # always skipped and caught by check_no_test_always_skipped.
        pytest.skip("Unknown OpenBLAS threading layer.")

    assert threading_layer in expected_openblas_threading_layers


# skip test if not run in a azure pipelines job since it relies on a specific flexiblas
# installation.
@pytest.mark.skipif(
    "TF_BUILD" not in os.environ, reason="not running in azure pipelines"
)
def test_flexiblas():
    # Check that threadpool_info correctly recovers the FlexiBLAS backends.
    flexiblas_controller = ThreadpoolController().select(internal_api="flexiblas")

    if not flexiblas_controller:
        pytest.skip("requires FlexiBLAS.")
    flexiblas_controller = flexiblas_controller.lib_controllers[0]

    expected_backends = {"NETLIB", "OPENBLAS_CONDA"}
    expected_backends_loaded = {"OPENBLAS_CONDA"}
    expected_current_backend = "OPENBLAS_CONDA"  # set as default at build time

    flexiblas_backends = flexiblas_controller.available_backends
    flexiblas_backends_loaded = flexiblas_controller.loaded_backends
    current_backend = flexiblas_controller.current_backend

    assert set(flexiblas_backends) == expected_backends
    assert set(flexiblas_backends_loaded) == expected_backends_loaded
    assert current_backend == expected_current_backend


def test_flexiblas_switch_error():
    # Check that an error is raised when trying to switch to an invalid backend.
    flexiblas_controller = ThreadpoolController().select(internal_api="flexiblas")

    if not flexiblas_controller:
        pytest.skip("requires FlexiBLAS.")
    flexiblas_controller = flexiblas_controller.lib_controllers[0]

    with pytest.raises(RuntimeError, match="Failed to load backend"):
        flexiblas_controller.switch_backend("INVALID_BACKEND")


# skip test if not run in a azure pipelines job since it relies on a specific flexiblas
# installation.
@pytest.mark.skipif(
    "TF_BUILD" not in os.environ, reason="not running in azure pipelines"
)
def test_flexiblas_switch():
    # Check that the backend can be switched.
    controller = ThreadpoolController()
    fb_controller = controller.select(internal_api="flexiblas")

    if not fb_controller:
        pytest.skip("requires FlexiBLAS.")
    fb_controller = fb_controller.lib_controllers[0]

    # at first mkl is not loaded in the CI jobs where this test runs
    assert len(controller.select(internal_api="mkl").lib_controllers) == 0

    # at first, only "OPENBLAS_CONDA" is loaded
    assert fb_controller.current_backend == "OPENBLAS_CONDA"
    assert fb_controller.loaded_backends == ["OPENBLAS_CONDA"]

    fb_controller.switch_backend("NETLIB")
    assert fb_controller.current_backend == "NETLIB"
    assert fb_controller.loaded_backends == ["OPENBLAS_CONDA", "NETLIB"]

    ext = ".so" if sys.platform == "linux" else ".dylib"
    mkl_path = f"{os.getenv('CONDA_PREFIX')}/lib/libmkl_rt{ext}"
    fb_controller.switch_backend(mkl_path)
    assert fb_controller.current_backend == mkl_path
    assert fb_controller.loaded_backends == ["OPENBLAS_CONDA", "NETLIB", mkl_path]
    # switching the backend triggered a new search for loaded shared libs
    assert len(controller.select(internal_api="mkl").lib_controllers) == 1

    # switch back to default to avoid side effects
    fb_controller.switch_backend("OPENBLAS_CONDA")


def test_threadpool_controller_as_decorator():
    # Check that using the decorator can be nested and is restricted to the scope of
    # the decorated function.
    controller = ThreadpoolController()
    original_info = controller.info()

    if any(info["num_threads"] < 2 for info in original_info):
        pytest.skip("Test requires at least 2 CPUs on host machine")
    if not controller.select(user_api="blas"):
        pytest.skip("Requires a blas runtime.")

    def check_blas_num_threads(expected_num_threads):
        blas_controller = ThreadpoolController().select(user_api="blas")
        assert all(
            lib_controller.num_threads == expected_num_threads
            for lib_controller in blas_controller.lib_controllers
        )

    @controller.wrap(limits=1, user_api="blas")
    def outer_func():
        check_blas_num_threads(expected_num_threads=1)
        inner_func()
        check_blas_num_threads(expected_num_threads=1)

    @controller.wrap(limits=2, user_api="blas")
    def inner_func():
        check_blas_num_threads(expected_num_threads=2)

    outer_func()

    assert ThreadpoolController().info() == original_info


def test_custom_controller():
    # Check that a custom controller can be used to change the number of threads
    # used by a library.
    try:
        import tests._pyMylib  # noqa
    except:
        pytest.skip("requires my_thread_lib to be compiled")

    controller = ThreadpoolController()
    original_info = controller.info()

    mylib_controller = controller.select(user_api="my_threaded_lib")

    # my_threaded_lib has been found and there's 1 matching shared library
    assert len(mylib_controller.lib_controllers) == 1
    mylib_controller = mylib_controller.lib_controllers[0]

    # we linked against my_threaded_lib v2.0 and by default it uses 42 thread
    assert mylib_controller.version == "2.0"
    assert mylib_controller.num_threads == 42

    # my_threaded_lib exposes an additional info "some_attr":
    assert mylib_controller.info()["some_attr"] == "some_value"

    with controller.limit(limits=1, user_api="my_threaded_lib"):
        assert mylib_controller.num_threads == 1

    assert ThreadpoolController().info() == original_info
