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

    use_custom_blas = os.getenv("INSTALL_BLAS", False)
    libraries = [use_custom_blas] if use_custom_blas else []
    custom_suffix = "_custom" if use_custom_blas else ""
    filename = f"nested_prange_blas{custom_suffix}.pyx"

    ext_modules = [
        Extension(
            "nested_prange_blas",
            [filename],
            extra_compile_args=openmp_flag,
            extra_link_args=openmp_flag,
            libraries=libraries,
        )
    ]

    setup(
        name="_openmp_test_helper_nested_prange_blas",
        ext_modules=cythonize(
            ext_modules,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
            },
        ),
    )

finally:
    os.environ.update(original_environ)
