import os
import pytest
from glob import glob


def skip_func(msg):
    def test_func(*args, **kwargs):
        pytest.skip(msg)
    return test_func


# Path to shipped openblas for libraries such as numpy or scipy
libopenblas_patterns = []


# A decorator to run tests only when numpy is available
try:
    # make sure the mkl/blas are loaded for test_threadpool_limits
    import numpy as np
    np.dot(np.ones(1000), np.ones(1000))

    libopenblas_patterns.append(os.path.join(np.__path__[0], ".libs",
                                             "libopenblas*"))

except ImportError:
    pass


try:
    import scipy
    import scipy.linalg  # noqa: F401
    scipy.linalg.svd([[1, 2], [3, 4]])

    libopenblas_patterns.append(os.path.join(scipy.__path__[0], ".libs",
                                             "libopenblas*"))
except ImportError:
    scipy = None

libopenblas_paths = set(path for pattern in libopenblas_patterns
                        for path in glob(pattern))


# A decorator to run tests only when check_openmp_n_threads is available
try:
    from ._openmp_test_helper import check_openmp_num_threads   # noqa: F401

    def with_check_openmp_num_threads(func):
        """A decorator to skip tests if check_openmp_n_threads is not compiled.
        """
        return func

except ImportError:
    def with_check_openmp_num_threads(func):
        """A decorator to skip tests if check_openmp_n_threads is not compiled.
        """
        return skip_func('Test requires check_openmp_n_threads to be compiled')
