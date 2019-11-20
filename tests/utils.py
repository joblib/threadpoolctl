import os
from glob import glob


# Path to shipped openblas for libraries such as numpy or scipy
libopenblas_patterns = []


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


try:
    from ._openmp_test_helper import check_openmp_num_threads  # noqa: F401
    cython_extensions_compiled = True
except ImportError:
    cython_extensions_compiled = False
