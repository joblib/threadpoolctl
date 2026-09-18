import os
import sys
import sysconfig


def set_cc_variables(var_name="CC"):
    cc_var = os.environ.get(var_name)
    if cc_var is not None:
        os.environ["CC"] = cc_var
        if sys.platform == "darwin":
            os.environ["LDSHARED"] = cc_var + " -bundle -undefined dynamic_lookup"
        else:
            os.environ["LDSHARED"] = cc_var + " -shared"

    return cc_var


def get_openmp_flag():
    if sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform == "darwin" and "openmp" in os.getenv("CPPFLAGS", ""):
        return []
    return ["-fopenmp"]


def cython_compiler_directives(**extra):
    """Return Cython compiler directives for the OpenMP test helpers.

    On free-threaded CPython, mark the extensions as GIL-free so importing
    them does not re-enable the GIL (which would fail CI under ``-W error``).
    """
    directives = {
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
    }
    directives.update(extra)
    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        directives["freethreading_compatible"] = True
    return directives
