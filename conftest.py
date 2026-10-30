import pytest

collect_ignore = [
    "tests/empirical_scope_observation.py",
    "tests/_openmp_test_helper.py",
    "tests/_limit_blas.py",
    "tests/_dl_iterate_phdr_deadlock.py",
    "tests/_get_cached_controller.py",
]


def _ensure_openblas_thread_count_is_restorable():
    """Replace an unset non-OpenMP OpenBLAS count with the processor count.

    ``get_num_threads() == 0`` means OpenBLAS has not chosen an active count.
    ``set_num_threads(0)`` does not return to that state: a non-positive value
    means the number of worker threads already created, and that number only
    grows. A positive count can be saved and restored.
    """
    from threadpoolctl import OpenBLASController, ThreadpoolController

    controller = ThreadpoolController().select(internal_api="openblas")
    if controller:
        for lib in controller.lib_controllers:
            if lib.threading_layer == "openmp" or lib.get_num_threads() != 0:
                continue
            get_num_procs = lib._get_symbol("openblas_get_num_procs")
            nprocs = get_num_procs() if get_num_procs is not None else 1
            if nprocs >= 1:
                lib.set_num_threads(nprocs)
    OpenBLASController.clear_controlling_thread(force=True)


@pytest.fixture(autouse=True)
def reset_openblas_controlling_thread():
    _ensure_openblas_thread_count_is_restorable()
    try:
        yield
    finally:
        _ensure_openblas_thread_count_is_restorable()
