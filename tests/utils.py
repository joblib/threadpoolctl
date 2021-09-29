import os
import json
import sys
import threadpoolctl
from glob import glob
from os.path import dirname, normpath
from subprocess import check_output


# Path to shipped openblas for libraries such as numpy or scipy
libopenblas_patterns = []


try:
    # make sure the mkl/blas are loaded for test_threadpool_limits
    import numpy as np

    np.dot(np.ones(1000), np.ones(1000))

    libopenblas_patterns.append(os.path.join(np.__path__[0], ".libs", "libopenblas*"))
except ImportError:
    pass


try:
    import scipy
    import scipy.linalg  # noqa: F401

    scipy.linalg.svd([[1, 2], [3, 4]])

    libopenblas_patterns.append(
        os.path.join(scipy.__path__[0], ".libs", "libopenblas*")
    )
except ImportError:
    scipy = None

libopenblas_paths = set(
    path for pattern in libopenblas_patterns for path in glob(pattern)
)


try:
    import tests._openmp_test_helper.openmp_helpers_inner  # noqa: F401

    cython_extensions_compiled = True
except ImportError:
    cython_extensions_compiled = False


def threadpool_info_from_subprocess(module):
    """Utility to call threadpool_info in a subprocess

    `module` is imported before calling threadpool_info
    """
    # set PYTHONPATH to import from non sub-modules
    path1 = normpath(dirname(threadpoolctl.__file__))
    path2 = os.path.join(path1, "tests", "_openmp_test_helper")
    pythonpath = os.pathsep.join([path1, path2])
    env = os.environ.copy()
    try:
        env["PYTHONPATH"] = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
    except KeyError:
        env["PYTHONPATH"] = pythonpath

    cmd = [sys.executable, "-m", "threadpoolctl", "-i", module]
    out = check_output(cmd, env=env).decode("utf-8")
    return json.loads(out)


def select(info, **kwargs):
    """Select a subset of the list of library info matching the request"""
    # It's just a utility function to avoid repeating the pattern
    # [lib_info for lib_info in info if lib_info["<key>"] == key]
    for key, vals in kwargs.items():
        kwargs[key] = [vals] if not isinstance(vals, list) else vals

    selected_info = [
        lib_info
        for lib_info in info
        if any(lib_info.get(key, None) in vals for key, vals in kwargs.items())
    ]

    return selected_info
