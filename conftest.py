import pytest
import warnings


def pytest_addoption(parser):
    parser.addoption("--openblas-present", action='store_true',
                     help="Fail test_limit_openblas_threads if BLAS is not "
                     "found")
    parser.addoption("--mkl-present", action='store_true',
                     help="Fail test_limit_mkl_threads if MKL is not "
                     "found")


@pytest.fixture
def openblas_present(request):
    """Fail the test if OpenBLAS is not found"""
    return request.config.getoption("--openblas-present")


@pytest.fixture
def mkl_present(request):
    """Fail the test if MKL is not found"""
    return request.config.getoption("--mkl-present")


def pytest_configure(config):
    """Verify the environment for testing threadpoolctl"""
    warnings.simplefilter('always')

    # When using this option, make sure numpy is accessible
    if config.getoption("--mkl-present"):
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ImportError("Need 'numpy' with option --mkl-present")
