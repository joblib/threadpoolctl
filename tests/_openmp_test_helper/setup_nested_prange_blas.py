import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from build_utils import set_cc_variables
from build_utils import get_openmp_flag


original_environ = os.environ.copy()
try:
    set_cc_variables("CC_OUTER_LOOP")
    openmp_flag = get_openmp_flag()

    ext_modules = [
        Extension(
            "nested_prange_blas",
            ["nested_prange_blas.pyx"],
            extra_compile_args=openmp_flag,
            extra_link_args=openmp_flag
            )
    ]

    setup(
        name='_openmp_test_helper_nested_prange_blas',
        ext_modules=cythonize(
            ext_modules,
            compiler_directives={'language_level': 3,
                                 'boundscheck': False,
                                 'wraparound': False})
    )

finally:
    os.environ.update(original_environ)
