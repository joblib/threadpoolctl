import os
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from build_utils import set_cc_variables


set_cc_variables("CC_OUTER_LOOP")

if sys.platform == "win32":
    openmp_flag = ["/openmp"]
elif sys.platform == "darwin" and 'openmp' in os.getenv('CPPFLAGS', ''):
    openmp_flag = []
else:
    openmp_flag = ["-fopenmp"]

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
        language_level=3,
        compiler_directives={'language_level': 3,
                             'boundscheck': False,
                             'wraparound': False})
)
