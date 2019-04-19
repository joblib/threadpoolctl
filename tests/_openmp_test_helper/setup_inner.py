import os
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from build_utils import set_cc_variables


original_environ = os.environ.copy()
try:
    # Make it possible to compile the 2 OpenMP enabled Cython extensions
    # with different compilers and therefore different OpenMP runtimes.
    inner_loop_cc_var = set_cc_variables("CC_INNER_LOOP")

    if sys.platform == "win32":
        openmp_flag = ["/openmp"]
    elif sys.platform == "darwin" and 'openmp' in os.getenv('CPPFLAGS', ''):
        openmp_flag = []
    else:
        openmp_flag = ["-fopenmp"]

    ext_modules = [
        Extension(
            "openmp_helpers_inner",
            ["openmp_helpers_inner.pyx"],
            extra_compile_args=openmp_flag,
            extra_link_args=openmp_flag
            )
    ]

    setup(
        name='_openmp_test_helper_inner',
        ext_modules=cythonize(
            ext_modules,
            language_level=3,
            compiler_directives={'language_level': 3,
                                 'boundscheck': False,
                                 'wraparound': False},
            compile_time_env={"CC_INNER_LOOP": inner_loop_cc_var or "unknown"})
    )

finally:
    os.environ.update(original_environ)
