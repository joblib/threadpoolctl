import pytest

collect_ignore = [
    "tests/empirical_scope_observation.py",
    "tests/_openmp_test_helper.py",
    "tests/_limit_blas.py",
    "tests/_dl_iterate_phdr_deadlock.py",
]

@pytest.fixture(autouse=True)
def reset_openblas_controlling_thread():
    from threadpoolctl import OpenBLASController
    try:
        yield
    finally:
        OpenBLASController.clear_controlling_thread(force=True)
