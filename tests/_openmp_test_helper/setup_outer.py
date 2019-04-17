import os
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


original_environ = os.environ.copy()
try:
    # Make it possible to compile the 2 OpenMP enabled Cython extensions
    # with different compilers and therefore different OpenMP runtimes.
    outer_loop_cc_var = os.environ.get("CC_OUTER_LOOP")
    if outer_loop_cc_var is not None:
        os.environ["CC"] = outer_loop_cc_var
        os.environ["LDSHARED"] = outer_loop_cc_var + " -shared"

    if sys.platform == "win32":
        extra_compile_args = ["/openmp"]
        extra_link_args = None
    else:
        extra_compile_args = ["-fopenmp"]
        extra_link_args = ["-fopenmp"]

    ext_modules = [
        Extension(
            "openmp_helpers_outer",
            ["openmp_helpers_outer.pyx"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
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
