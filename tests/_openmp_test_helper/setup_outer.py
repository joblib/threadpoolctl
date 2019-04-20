import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from build_utils import set_cc_variables
from build_utils import get_openmp_flag


original_environ = os.environ.copy()
try:
    # Make it possible to compile the 2 OpenMP enabled Cython extensions
    # with different compilers and therefore different OpenMP runtimes.
    outer_loop_cc_var = set_cc_variables("CC_OUTER_LOOP")
    openmp_flag = get_openmp_flag()

    ext_modules = [
        Extension(
            "openmp_helpers_outer",
            ["openmp_helpers_outer.pyx"],
            extra_compile_args=openmp_flag,
            extra_link_args=openmp_flag
            )
    ]

    setup(
        name="_openmp_test_helper_outer",
        ext_modules=cythonize(
            ext_modules,
            compiler_directives={'language_level': 3,
                                 'boundscheck': False,
                                 'wraparound': False},
            compile_time_env={"CC_OUTER_LOOP": outer_loop_cc_var or "unknown"})
    )

finally:
    os.environ.update(original_environ)
