import json
import os
import sys
import ctypes
import shutil
import threadpoolctl
from glob import glob
from os.path import dirname, normpath
from pathlib import Path
from subprocess import check_output

# Path to shipped openblas for libraries such as numpy or scipy
libopenblas_patterns = []


try:
    # make sure the mkl/blas are loaded for test_threadpool_limits
    import numpy as np

    np.dot(np.ones(1000), np.ones(1000))

    libopenblas_patterns.append(os.path.join(np.__path__[0], ".libs", "libopenblas*"))
    numpy_site_packages = os.path.dirname(np.__path__[0])
    libopenblas_patterns.append(
        os.path.join(numpy_site_packages, "numpy.libs", "libscipy_openblas*.dll")
    )
    if sys.platform == "win32":
        libopenblas_patterns.append(
            os.path.join(sys.prefix, "Library", "bin", "openblas*.dll")
        )
        libopenblas_patterns.append(
            os.path.join(sys.prefix, "Library", "bin", "libopenblas*.dll")
        )
except ImportError:
    pass


try:
    import scipy
    import scipy.linalg  # noqa: F401

    scipy.linalg.svd([[1, 2], [3, 4]])

    libopenblas_patterns.append(
        os.path.join(scipy.__path__[0], ".libs", "libopenblas*")
    )
    scipy_site_packages = os.path.dirname(scipy.__path__[0])
    libopenblas_patterns.append(
        os.path.join(scipy_site_packages, "scipy.libs", "libscipy_openblas*.dll")
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


try:
    from tests._openmp_test_helper.nested_prange_blas import check_nested_prange_blas
except ImportError:
    check_nested_prange_blas = None


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


def get_openblas_dll_path():
    """Return a path to an OpenBLAS DLL that can be copied for Windows tests."""
    if libopenblas_paths:
        return sorted(
            libopenblas_paths,
            key=lambda path: (
                0
                if "libscipy_openblas" in os.path.basename(path).lower()
                else 1
                if "libopenblas" in os.path.basename(path).lower()
                else 2
            ),
        )[0]

    from threadpoolctl import ThreadpoolController

    controllers = ThreadpoolController().select(internal_api="openblas").lib_controllers
    if not controllers:
        return None

    filepath = controllers[0].filepath
    if os.path.isfile(filepath):
        return filepath
    return None


def normalize_windows_path(path):
    """Normalize Windows paths for stable comparisons in tests."""
    path = os.path.abspath(str(path))
    if path.startswith("\\\\?\\"):
        path = path[4:]
        if path.startswith("UNC\\"):
            path = "\\\\" + path[4:]
    return os.path.normcase(os.path.normpath(path))


LONG_PATH_OPENBLAS_DLL = "libopenblas_long_path_test.dll"
TRUNCATED_PATH_OPENBLAS_DLL = "libopenblas_path_too_long.dll"


def make_long_windows_path(base_dir, filename, min_length=261):
    """Nest padded directories under base_dir until base/.../filename >= min_length."""
    padding_segments = ["a" * 100, "b" * 100, "c" * 100, "d" * 100]
    current = Path(base_dir)
    segment_index = 0
    target = current / filename

    while len(str(target)) < min_length:
        current = current / padding_segments[segment_index % len(padding_segments)]
        segment_index += 1
        target = current / filename

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def to_extended_windows_path(path):
    """Return an extended-length path for Windows APIs (> MAX_PATH)."""
    path = os.path.abspath(str(path))
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path


def copy_and_load_dll(src_dll, destination):
    """Copy a DLL to destination and load it in the current process."""
    shutil.copy2(src_dll, to_extended_windows_path(destination))
    ctypes.CDLL(to_extended_windows_path(destination))
