"""threadpoolctl

This module provides utilities to introspect native libraries that relies on
thread pools (notably BLAS and OpenMP implementations) and dynamically set the
maximal number of threads they can use.
"""

# License: BSD 3-Clause

# The code to introspect dynamically loaded libraries on POSIX systems is
# adapted from code by Intel developer @anton-malakhov available at
# https://github.com/IntelPython/smp (Copyright (c) 2017, Intel Corporation)
# and also published under the BSD 3-Clause license

from __future__ import annotations

import ctypes
import itertools
import os
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from threading import Lock, Thread, current_thread
from threading import local as thread_local
from typing import Literal, TypeVar, final

# ctypes.util is not imported on Linux: on CPython 3.14 it allocates a
# process-lifetime CFUNCTYPE callback that is not fork-safe with some libffi
# builds (#225). dllist also uses dl_iterate_phdr internally (#239), which we
# already avoid on Linux via /proc/self/maps on older versions of Python.
dllist = None
if (
    sys.platform != "emscripten"
    and (
        sys.platform != "linux"
        # Python 3.15 doesn't have the CFUNCTYPE anymore:
        or sys.version_info[:2] >= (3, 15)
    )
):
    try:
        from ctypes.util import dllist
    except ImportError:
        # CPython before 3.14 does not provide dll inspection.
        dllist = None

# Ensure we don't get spurious cache invalidation the first time
# ThreadpoolController is created, as the Windows dll listing methods import
# from there:
if sys.platform == "win32":
    from ctypes import wintypes

    del wintypes


__version__ = "3.8.0.dev0"
__all__ = [
    "SHARED_POOL_CONTROL_OWNERSHIP",
    "LibController",
    "ThreadpoolController",
    "get_cached_controller",
    "register",
    "reset_cached_controller",
    "threadpool_info",
    "threadpool_limits",
]


# One can get runtime errors or even segfaults due to multiple OpenMP libraries
# loaded simultaneously which can happen easily in Python when importing and
# using compiled extensions built with different compilers and therefore
# different OpenMP runtimes in the same program. In particular libiomp (used by
# Intel ICC) and libomp used by clang/llvm tend to crash. This can happen for
# instance when calling BLAS inside a prange. Setting the following environment
# variable allows multiple OpenMP libraries to be loaded. It should not degrade
# performances since we manually take care of potential over-subscription
# performance issues, in sections of the code where nested OpenMP loops can
# happen, by dynamically reconfiguring the inner OpenMP runtime to temporarily
# disable it while under the scope of the outer OpenMP parallel section.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Hard limit for Windows library paths resolved through GetModuleFileNameExW.
# Kept below unlimited for security reasons; see CHANGES.md and PR #189.
_WINDOWS_MAX_LIBRARY_PATH_LENGTH = 2600

# Structure to cast the info on dynamically loaded library. See
# https://linux.die.net/man/3/dl_iterate_phdr for more details.
_SYSTEM_UINT = ctypes.c_uint64 if sys.maxsize > 2**32 else ctypes.c_uint32
_SYSTEM_UINT_HALF = ctypes.c_uint32 if sys.maxsize > 2**32 else ctypes.c_uint16


_HAS_PROCFS = (
    # Only available on Linux:
    sys.platform == "linux"
    # Make sure /proc is mounted:
    and os.path.exists("/proc/self")
)


class _dl_phdr_info(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr", _SYSTEM_UINT),  # Base address of object
        ("dlpi_name", ctypes.c_char_p),  # path to the library
        ("dlpi_phdr", ctypes.c_void_p),  # pointer on dlpi_headers
        ("dlpi_phnum", _SYSTEM_UINT_HALF),  # number of elements in dlpi_phdr
    ]


# The RTLD_NOLOAD flag for loading shared libraries is not defined on Windows.
try:
    _RTLD_NOLOAD = os.RTLD_NOLOAD
except AttributeError:
    _RTLD_NOLOAD = ctypes.DEFAULT_MODE


# What scope does the API affect?
_ThreadLimitScope = Literal[
    # Using the API sets a limit only on the current thread.
    "current_thread",
    # Using the API sets a limit for every thread in the process; whether or
    # not it's a shared process-wide pool or per-thread limit needs to be
    # determined some other way.
    "process",
    # Something else, unexpected; perhaps another variant, perhaps information
    # can't be determined under the current configuration.
    "unknown",
]


def _determine_thread_limit_scope(
    get_n_threads: Callable[[], int], set_n_threads: Callable[[int], None]
) -> _ThreadLimitScope:
    """
    Run some experiments to determine the scope of the given get/set API.

    This function might not work if you only have one core available.

    This function might not work if you set a limit on a library with an
    environment variable.

    The function works by changing the number of threads in loaded controllers,
    which can be a process-wide change. As such, it is not always thread-safe.
    An attempt will be made to restore all settings to their previous state,
    but the result may be subtly different, e.g. if "unset" has different
    semantics than "set to the default returned value".
    """
    previous = get_n_threads()

    # Some plausible constraints we need to keep in mind:
    #
    # 1. The API might not allow setting more than the number of (available, or
    #    physical) cores.
    # 2. Some hard limit on number of threads.
    try:
        # Choose a desired number of threads that is different than the current
        # number, and hopefully achievable under the current configuration:
        if previous < 2:
            expected = 2
        else:
            # It's 2 or more, so shrink it slightly:
            expected = previous - 1

        thread_result = []

        def get_and_set() -> None:
            set_n_threads(expected)
            thread_result.append(get_n_threads())

        thread = Thread(target=get_and_set)
        thread.start()
        thread.join()
        OpenBLASController.clear_controlling_thread(force=True)

        # First, getting in the same thread as a set should always give same
        # number, if it's a number in a reasonable range. A possible exception
        # fo failing this is if the number of thread is limited by available
        # CPU, and only one CPU is available. In that case we can't empirically
        # determine how the API works. We try to not reach that point here, but
        # you can imagine a thread pool implementation that is aware of
        # cgroups, in which case a Docker container limited to one core will
        # pass the safety check at the start of the function. Perhaps
        # cpu_count() from loky should be moved into this package...
        if thread_result != [expected]:
            return "unknown"

        # Now, check this thread:
        if get_n_threads() == expected:
            # Setting modified this thread's results too:
            return "process"
        elif get_n_threads() == previous:
            # Setting modified the other thread, but not this one:
            return "current_thread"
        else:
            # No idea what's going on:
            return "unknown"
    finally:
        set_n_threads(previous)
        OpenBLASController.clear_controlling_thread(force=True)


class LibController(ABC):
    """Abstract base class for the individual library controllers

    A library controller must expose the following class attributes:
        - user_api : str
            Usually the name of the library or generic specification the library
            implements, e.g. "blas" is a specification with different implementations.
        - internal_api : str
            Usually the name of the library or concrete implementation of some
            specification, e.g. "openblas" is an implementation of the "blas"
            specification.
        - filename_prefixes : tuple
            Possible prefixes of the shared library's filename that allow to
            identify the library. e.g. "libopenblas" for libopenblas.so.

    and implement the following methods: `get_num_threads`, `set_num_threads` and
    `get_version`.

    Threadpoolctl loops through all the loaded shared libraries and tries to match
    the filename of each library with the `filename_prefixes`. If a match is found, a
    controller is instantiated and a handler to the library is stored in the `dynlib`
    attribute as a `ctypes.CDLL` object. It can be used to access the necessary symbols
    of the shared library to implement the above methods.

    The following information will be exposed in the info dictionary:
      - user_api : standardized API, if any, or a copy of internal_api.
      - internal_api : implementation-specific API.
      - num_threads : the current thread limit.
      - prefix : prefix of the shared library's filename.
      - filepath : path to the loaded shared library.
      - version : version of the library (if available).

    In addition, each library controller may expose internal API specific entries. They
    must be set as attributes in the `set_additional_attributes` method.
    """

    @final
    def __init__(self, *, filepath=None, prefix=None, parent=None):
        """This is not meant to be overriden by subclasses."""
        self.parent = parent
        self.prefix = prefix
        self.filepath = filepath
        self.dynlib = _CDLL_CACHE.get_cdll(filepath)
        self._symbol_prefix, self._symbol_suffix = self._find_affixes()
        self.version = self.get_version()
        self.set_additional_attributes()

    def info(self, debugging_info: bool = False):
        """Return relevant info wrapped in a dict.

        Parameters
        ----------
        debugging_info : bool

            Include extra fields which require more intrusive actions to
            obtain, and which can't always reliably be determined.
        """
        hidden_attrs = ("dynlib", "parent", "_symbol_prefix", "_symbol_suffix")
        result = {
            "user_api": self.user_api,
            "internal_api": self.internal_api,
            "num_threads": self.num_threads,
            **{k: v for k, v in vars(self).items() if k not in hidden_attrs},
        }
        if debugging_info:
            result["thread_limit_scope"] = _determine_thread_limit_scope(
                self.get_num_threads, self.set_num_threads
            )
        return result

    def set_additional_attributes(self):
        """Set additional attributes meant to be exposed in the info dict"""

    @property
    def num_threads(self):
        """Exposes the current thread limit as a dynamic property

        This is not meant to be used or overriden by subclasses.
        """
        return self.get_num_threads()

    @abstractmethod
    def get_num_threads(self):
        """Return the maximum number of threads available to use"""

    @abstractmethod
    def set_num_threads(self, num_threads):
        """Set the maximum number of threads to use

        When possible, implementations of this method should choose a thread
        limiting API that only applies to the current thread.
        """

    @abstractmethod
    def get_version(self):
        """Return the version of the shared library"""

    def _find_affixes(self):
        """Return the affixes for the symbols of the shared library"""
        return "", ""

    def _get_symbol(self, name):
        """Return the symbol of the shared library accounding for the affixes"""
        return getattr(
            self.dynlib, f"{self._symbol_prefix}{name}{self._symbol_suffix}", None
        )

    def _has_shared_pool(self):
        """Whether this library's thread pool is shared by all Python threads.

        When True, only one thread at a time can change the pool size, and
        ``threadpool_limits`` scopes claim control of it on that library.
        Controllers with a process-wide pool inherit ``_SharedPoolControllerMixin``,
        which reports True. Override when only some builds share a pool.
        """
        return False


# Internal marker for missing cache lookups:
_MISSING = object()


_T = TypeVar("_T")


@dataclass
class _CDLLCache:
    """
    Cache CDLL instances and their associated method results, as well as which
    ``LibController`` subclass to use for a given shared library file path.
    """

    _controller_cache: dict[str, type[LibController] | None] = field(
        default_factory=dict
    )

    _cdll_cache: dict[str, ctypes.CDLL] = field(default_factory=dict)

    _method_result_cache: dict[tuple[type[LibController], str, ctypes.CDLL], object] = (
        field(default_factory=dict)
    )

    def _check_prefix(
        self, library_basename: str, filename_prefixes: list[str]
    ) -> str | None:
        """Return the prefix library_basename starts with

        Return None if none matches.
        """
        for prefix in filename_prefixes:
            if library_basename.startswith(prefix):
                return prefix
        return None

    def create_controller(
        self, filepath: str, parent: ThreadpoolController
    ) -> LibController | None:
        """
        Create the associated controller, if there is one, relying on cached
        info for the given ``filepath``.
        """
        result = self._controller_cache.get(filepath, _MISSING)
        if result is not _MISSING:
            controller_class, filepath, prefix = result
            if controller_class is None:
                return None
            return controller_class(filepath=filepath, prefix=prefix, parent=parent)

        # It's not in the cache, so proceed to actually search for it.

        # Required to resolve symlinks
        original_filepath = filepath
        filepath = _realpath(filepath)
        # `lower` required to take account of OpenMP dll case on Windows
        # (vcomp, VCOMP, Vcomp, ...)
        filename = os.path.basename(filepath).lower()

        # Loop through supported libraries to find if this filename corresponds
        # to a supported one.
        for controller_class in _ALL_CONTROLLERS:
            # check if filename matches a supported prefix
            prefix = self._check_prefix(filename, controller_class.filename_prefixes)

            # filename does not match any of the prefixes of the candidate
            # library. move to next library.
            if prefix is None:
                continue

            # Legacy workaround for BLAS libraries that conda-forge used to expose
            # on Windows as libblas.dll, disambiguated via implementation-specific
            # symbols. Current conda-forge stacks no longer load libblas.dll (e.g.
            # MKL is exposed as mkl_rt.<version>.dll instead), so this path is
            # kept for older installs but cannot be exercised in today's CI.
            if prefix == "libblas":
                if filename.endswith(".dll"):
                    libblas = ctypes.CDLL(filepath, _RTLD_NOLOAD)
                    if not any(
                        hasattr(libblas, func)
                        for func in controller_class.check_symbols
                    ):
                        continue
                else:
                    # Non-Windows libblas DSOs (e.g. from openblas) lack the symbols
                    # needed to instantiate a controller and would duplicate entries.
                    continue

            # filename matches a prefix. Now we check if the library has the symbols we
            # are looking for. If none of the symbols exists, it's very likely not the
            # expected library (e.g. a library having a common prefix with one of the
            # our supported libraries). Otherwise, create and store the library
            # controller.
            try:
                lib_controller = controller_class(
                    filepath=filepath, prefix=prefix, parent=parent
                )
            except OSError:
                # Probably because we couldn't load the filepath as a CDLL.
                continue

            if not hasattr(controller_class, "check_symbols") or any(
                hasattr(lib_controller.dynlib, func)
                for func in controller_class.check_symbols
            ):
                self._controller_cache[original_filepath] = (
                    controller_class,
                    filepath,
                    prefix,
                )
                return lib_controller

        # Didn't find any matching libraries.
        self._controller_cache[original_filepath] = (None, "", "")
        return None

    def get_cdll(self, filepath: str) -> ctypes.CDLL:
        """
        Get the ``CDLL`` for a path, loading if necessary, using a cached
        version if it was already loaded.
        """
        result = self._cdll_cache.get(filepath, _MISSING)
        if result is _MISSING:
            result = ctypes.CDLL(filepath, mode=_RTLD_NOLOAD)
            self._cdll_cache[filepath] = result
        return result

    def cache_method_on_dynlib(
        self, method: Callable[[LibController], _T]
    ) -> Callable[[LibController], _T]:
        """
        Caching decorator for idempotent read-only methods of
        ``LibController``, with the cache shared across instances that have the
        same ``CDLL`` instance.
        """
        cache = self._method_result_cache
        name = method.__name__

        @wraps(method)
        def wrapper(self):
            key = (self.__class__, name, self.dynlib)
            result = cache.get(key, _MISSING)
            if result is _MISSING:
                result = method(self)
                cache[key] = result
            return result

        return wrapper


_CDLL_CACHE = _CDLLCache()


@dataclass
class _SharedPoolState:
    """Who may change one shared thread pool."""

    controlling_thread: Thread | None = None
    # Set when control is taken outside a limiter scope (direct set_num_threads).
    sticky: bool = False
    lock: Lock = field(default_factory=Lock)
    # Controller class for this pool, so a clear of one class leaves the others.
    controller_class: type | None = None


# Process-wide claims on shared thread pools, keyed by filepath. A controller opts in by
# inheriting _SharedPoolControllerMixin. Separate copies of a library, for example
# NumPy's and SciPy's OpenBLAS, have separate pools, so each copy has its own record and
# its own controlling thread.
#
# Lock order: take _SHARED_POOL_STATES_LOCK before state.lock. Do not call sync() or
# count_for() while holding state.lock. Both take the states lock, and releasing a
# claim takes the states lock before the pool lock.
_SHARED_POOL_STATES: dict[str, _SharedPoolState] = {}
_SHARED_POOL_STATES_LOCK = Lock()
# Bumped per controller class by a forced clear. A thread drops a class's counts when
# its stored generation for that class does not match.
_shared_pool_generations: dict[type, int] = {}


def _get_shared_pool_state(filepath: str, controller_class: type) -> _SharedPoolState:
    with _SHARED_POOL_STATES_LOCK:
        state = _SHARED_POOL_STATES.get(filepath)
        if state is None:
            state = _SharedPoolState(controller_class=controller_class)
            _SHARED_POOL_STATES[filepath] = state
        elif state.controller_class is None:
            state.controller_class = controller_class
        return state


def _snapshot_shared_pool_states() -> list[_SharedPoolState]:
    with _SHARED_POOL_STATES_LOCK:
        return list(_SHARED_POOL_STATES.values())


def _other_live_thread_holds(holder, this) -> bool:
    """Whether ``holder`` is a different thread that still owns the pool."""
    return holder is not None and holder is not this and holder.is_alive()


def _iter_claimable_shared_pools(controllers):
    """Yield shared-pool libraries in ``controllers`` this thread can claim.

    One library per filepath. A pool another live thread already holds is skipped.
    """
    this = current_thread()
    seen = set()
    for lib in controllers:
        if not lib._has_shared_pool():
            continue
        if lib.filepath in seen:
            continue
        seen.add(lib.filepath)
        holder = lib._shared_pool_state().controlling_thread
        if _other_live_thread_holds(holder, this):
            continue
        yield lib


def _release_shared_pool_claim(filepath: str):
    """Drop this thread's non-sticky claim on one shared pool.

    A sticky claim was taken by a direct ``set_num_threads()`` while no limiter
    scope was active for that library. Scoped exits must leave it in place.
    """
    with _SHARED_POOL_STATES_LOCK:
        state = _SHARED_POOL_STATES.get(filepath)
    if state is None:
        return
    this = current_thread()
    with state.lock:
        if state.controlling_thread is this and not state.sticky:
            state.controlling_thread = None


def _clear_shared_pool_claims(controller_class: type, force: bool = False):
    """Drop claims on shared pools of one controller class.

    Pools of other classes are left untouched, including when ``force`` is
    true. A forced clear also drops that class's counts on other threads the
    next time they sync.
    """
    this = current_thread()
    cleared_this_thread = False
    for state in _snapshot_shared_pool_states():
        if state.controller_class is not controller_class:
            continue
        with state.lock:
            if force or state.controlling_thread is this:
                if state.controlling_thread is this:
                    cleared_this_thread = True
                state.controlling_thread = None
                state.sticky = False
    if force:
        with _SHARED_POOL_STATES_LOCK:
            generation = _shared_pool_generations.get(controller_class, 0) + 1
            _shared_pool_generations[controller_class] = generation
    if force or cleared_this_thread:
        SHARED_POOL_CONTROL_OWNERSHIP.reset(controller_class)


class _SharedPoolControllerMixin:
    """Controlling-thread claims for a library with a process-wide thread pool.

    Inherit this next to ``LibController`` and implement ``_real_set_num_threads``.
    ``set_num_threads`` should call ``_set_shared_pool_num_threads`` when the
    loaded library shares one pool across Python threads. Override
    ``_has_shared_pool`` when that depends on how the library was built.
    ``clear_controlling_thread`` drops claims of this controller class only.
    """

    def _has_shared_pool(self):
        """Whether this loaded library currently uses a process-wide pool."""
        return True

    def _shared_pool_state(self) -> _SharedPoolState:
        """Return the process-wide claim record for this library's shared pool."""
        return _get_shared_pool_state(self.filepath, type(self))

    @classmethod
    def clear_controlling_thread(cls, force: bool = False):
        """Reset which thread can limit this controller's shared pools.

        Drops the claim on every library of this controller class that the
        current thread controls, and resets this thread's ownership counts for
        that class so a later context manager can release a claim of its own.
        Shared pools of other controller classes are left in place. It's up to
        the caller to ensure the current controlling thread is done.

        By default, only the controlling thread has an effect. Any thread can
        clear every library of this class by setting ``force=True``.
        """
        _clear_shared_pool_claims(cls, force)

    def _set_shared_pool_num_threads(self, num_threads):
        """Set the pool size if this thread holds or can take the claim.

        Another live thread's call does nothing. A finished controlling thread
        no longer holds the claim. A direct call, while this thread's ownership
        count is still 0, is sticky: a later context manager must not drop it.
        A call made while a limiter scope is active is released with that scope.
        """
        SHARED_POOL_CONTROL_OWNERSHIP.sync()
        this_thread = current_thread()
        state = self._shared_pool_state()
        controlling_thread = state.controlling_thread
        if controlling_thread is None or not controlling_thread.is_alive():
            # Read the count before state.lock. count_for() takes the states lock.
            count = SHARED_POOL_CONTROL_OWNERSHIP.count_for(self.filepath)
            with state.lock:
                holder = state.controlling_thread
                if holder is None or not holder.is_alive():
                    state.controlling_thread = this_thread
                    state.sticky = count == 0
            # Retry now that the claim attempt is done. If this thread won, the
            # next call reaches _real_set_num_threads. If another live thread won,
            # the next call does nothing.
            return self._set_shared_pool_num_threads(num_threads)
        elif controlling_thread is this_thread:
            return self._real_set_num_threads(num_threads)
        else:
            return


class OpenBLASController(_SharedPoolControllerMixin, LibController):
    """Controller class for OpenBLAS.

    When OpenBLAS uses the OpenMP threading layer, each thread that calls into
    OpenBLAS gets its own OpenBLAS thread pool, and correspondingly the limit
    set by ``threadpoolctl`` is per-thread.  Each Python thread can therefore
    set its own limit without consideration for other Python threads.

    In contrast, when the OpenMP threading layer is not used, there is a
    process-wide shared thread pool, so having multiple Python threads change
    the limit at the same time makes no sense. Each loaded OpenBLAS library
    has its own pool and its own controlling thread. Therefore:

        1. Thread pool size can only be set by a single thread, the first
           thread to call ``set_num_threads()`` on that library. In practice
           this will usually be the main thread.

        2. Other threads silently do nothing. Once the controlling thread has
           finished, the next ``set_num_threads()`` can take the claim.

        3. ``clear_controlling_thread()`` resets the controlling thread. By
           default only that thread can do so. ``force=True`` lets any thread
           clear every OpenBLAS library.

    A plain ``threadpool_limits(...)`` call, and a direct ``set_num_threads()``,
    keep control until ``clear_controlling_thread()``. ``with
    threadpool_limits()`` and ``threadpool_limits.wrap()`` release control when
    the outermost scope that took the claim exits, including when those scopes
    are nested. A scope that does not touch that OpenBLAS library, and a scope
    entered while a plain call or a direct ``set_num_threads()`` is still
    holding the claim, does not release it.
    """

    user_api = "blas"
    internal_api = "openblas"
    filename_prefixes = (
        "libopenblas",
        "libblas",  # legacy conda-forge Windows shim, see _make_controller_from_path
        "libscipy_openblas",
        "openblas",  # Windows conda package use openblas.dll
    )

    _symbol_prefixes = ("", "scipy_")
    _symbol_suffixes = ("", "64_", "_64")

    # All variations of "openblas_get_num_threads", accounting for the affixes
    check_symbols = tuple(
        f"{prefix}openblas_get_num_threads{suffix}"
        for prefix, suffix in itertools.product(_symbol_prefixes, _symbol_suffixes)
    )

    @_CDLL_CACHE.cache_method_on_dynlib
    def _find_affixes(self):
        for prefix, suffix in itertools.product(
            self._symbol_prefixes, self._symbol_suffixes
        ):
            if hasattr(self.dynlib, f"{prefix}openblas_get_num_threads{suffix}"):
                return prefix, suffix

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        # See discussion in set_num_threads for details:
        if self.threading_layer == "openmp":
            symbol = "omp_get_max_threads"
        else:
            symbol = "openblas_get_num_threads"
        get_num_threads_func = self._get_symbol(symbol)
        if get_num_threads_func is not None:
            return get_num_threads_func()
        return None

    def _has_shared_pool(self):
        return self.threading_layer != "openmp"

    def set_num_threads(self, num_threads):
        if self.threading_layer == "openmp":
            # Per-thread limits, so no need to be constraining:
            return self._real_set_num_threads(num_threads)
        return self._set_shared_pool_num_threads(num_threads)

    def _real_set_num_threads(self, num_threads):
        # The OpenBLAS limiting API is process-wide, and we want current thread
        # limit if possible. When OpenBLAS is backed by OpenMP, using the
        # OpenMP API allows for current thread limiting when OpenMP has that
        # behavior. That is the case for libgomp, libomp, and libiomp, what you
        # would find on Linux or macOS.
        #
        # On Windows the Visual C++ OpenMP API is process-wide, unfortunately,
        # though this may be fixed if the /openmp:llvm flag is used:
        # https://github.com/joblib/threadpoolctl/issues/230
        #
        # Also worth knowing that before v0.3.34, the OpenBLAS limiting API is
        # broken when using OpenMP threading:
        # https://github.com/OpenMathLib/OpenBLAS/issues/5806
        if self.threading_layer == "openmp":
            symbol = "omp_set_num_threads"
        else:
            symbol = "openblas_set_num_threads"
        set_num_threads_func = self._get_symbol(symbol)
        if set_num_threads_func is not None:
            return set_num_threads_func(num_threads)
        return None

    @_CDLL_CACHE.cache_method_on_dynlib
    def get_version(self):
        # None means OpenBLAS is not loaded or version < 0.3.4, since OpenBLAS
        # did not expose its version before that.
        get_version_func = self._get_symbol("openblas_get_config")
        if get_version_func is not None:
            get_version_func.restype = ctypes.c_char_p
            config = get_version_func().split()
            if config[0] == b"OpenBLAS":
                return config[1].decode("utf-8")
            return None
        return None

    @_CDLL_CACHE.cache_method_on_dynlib
    def _get_threading_layer(self):
        """Return the threading layer of OpenBLAS"""
        get_threading_layer_func = self._get_symbol("openblas_get_parallel")
        if get_threading_layer_func is not None:
            threading_layer = get_threading_layer_func()
            if threading_layer == 2:
                return "openmp"
            elif threading_layer == 1:
                return "pthreads"
            return "disabled"
        return "unknown"

    @_CDLL_CACHE.cache_method_on_dynlib
    def _get_architecture(self):
        """Return the architecture detected by OpenBLAS"""
        get_architecture_func = self._get_symbol("openblas_get_corename")
        if get_architecture_func is not None:
            get_architecture_func.restype = ctypes.c_char_p
            return get_architecture_func().decode("utf-8")
        return None


class _SharedPoolControlOwnership(thread_local):
    """Per-pool count of limiter scopes on this thread.

    A count is kept separately for each shared pool, keyed by library filepath.
    Hitting zero for a pool releases that pool's non-sticky controlling thread.
    A plain ``threadpool_limits(...)`` call increments and never decrements, so
    it holds its claim until ``clear_controlling_thread()``.

    ``increment(controller)`` counts every shared pool of ``controller`` that
    this thread can control. ``decrement(controller)`` counts those pools back
    down. ``controller`` defaults to ``get_cached_controller()``. Call
    ``increment()`` before ``set_num_threads`` and ``decrement()`` when that
    scope ends to manage the claim without a context manager. A forced clear
    drops counts for that controller class only.
    """

    def _counts(self) -> dict[str, int]:
        counts = getattr(self, "counts", None)
        if counts is None:
            counts = {}
            self.counts = counts
        return counts

    def _generations(self) -> dict[type, int]:
        generations = getattr(self, "generations", None)
        if generations is None:
            generations = {}
            self.generations = generations
        return generations

    def sync(self):
        """Drop counts for libraries another thread has force-cleared."""
        with _SHARED_POOL_STATES_LOCK:
            generations = dict(_shared_pool_generations)
            owned_filepaths = [
                (filepath, state.controller_class)
                for filepath, state in _SHARED_POOL_STATES.items()
                if state.controller_class is not None
            ]
        local = self._generations()
        stale = {
            controller_class
            for controller_class, generation in generations.items()
            if local.get(controller_class, 0) != generation
        }
        if not stale:
            return
        counts = self._counts()
        for filepath, controller_class in owned_filepaths:
            if controller_class in stale:
                counts.pop(filepath, None)
        for controller_class in stale:
            local[controller_class] = generations[controller_class]

    def reset(self, controller_class: type | None = None):
        """Forget scopes on this thread.

        With ``controller_class``, only that class. Without, every library.
        """
        if controller_class is None:
            with _SHARED_POOL_STATES_LOCK:
                generations = dict(_shared_pool_generations)
            self.counts = {}
            self.generations = generations
            return
        with _SHARED_POOL_STATES_LOCK:
            generation = _shared_pool_generations.get(controller_class, 0)
            filepaths = [
                filepath
                for filepath, state in _SHARED_POOL_STATES.items()
                if state.controller_class is controller_class
            ]
        counts = self._counts()
        for filepath in filepaths:
            counts.pop(filepath, None)
        self._generations()[controller_class] = generation

    def count_for(self, filepath: str) -> int:
        """Active scopes for one shared pool on this thread."""
        self.sync()
        return self._counts().get(filepath, 0)

    def increment(self, controller: ThreadpoolController | None = None):
        """Increment every shared pool of ``controller`` this thread can control.

        ``controller`` defaults to ``get_cached_controller()``. Call this
        before ``set_num_threads`` so the claim stays scoped.
        """
        if controller is None:
            controller = get_cached_controller()
        for lib in _iter_claimable_shared_pools(controller.lib_controllers):
            self._increment(lib.filepath)

    def _increment(self, filepath: str):
        """Count one more scope for a single shared pool."""
        self.sync()
        counts = self._counts()
        counts[filepath] = counts.get(filepath, 0) + 1

    def decrement(self, controller: ThreadpoolController | None = None):
        """Decrement pools of ``controller`` that this thread currently counts.

        ``controller`` defaults to ``get_cached_controller()``. A pool whose
        count hits zero releases its non-sticky claim. This closes a matching
        ``increment()`` for that controller.
        """
        if controller is None:
            controller = get_cached_controller()
        self.sync()
        counts = self._counts()
        seen = set()
        for lib in controller.lib_controllers:
            if not lib._has_shared_pool():
                continue
            if lib.filepath in seen or lib.filepath not in counts:
                continue
            seen.add(lib.filepath)
            self._decrement(lib.filepath)

    def _decrement(self, filepath: str):
        """Count one fewer scope for a single shared pool.

        Reaching zero releases that pool's non-sticky claim.
        """
        self.sync()
        counts = self._counts()
        current = counts.get(filepath, 0)
        if current <= 0:
            return
        current -= 1
        if current == 0:
            counts.pop(filepath, None)
            _release_shared_pool_claim(filepath)
        else:
            counts[filepath] = current


SHARED_POOL_CONTROL_OWNERSHIP = _SharedPoolControlOwnership()


class BLISController(LibController):
    """Controller class for BLIS"""

    user_api = "blas"
    internal_api = "blis"
    filename_prefixes = (
        "libblis",
        "libblas",
    )  # libblas: legacy conda-forge Windows shim
    check_symbols = (
        "bli_thread_get_num_threads",
        "bli_thread_set_num_threads",
        "bli_info_get_version_str",
        "bli_info_get_enable_openmp",
        "bli_info_get_enable_pthreads",
        "bli_arch_query_id",
        "bli_arch_string",
    )

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()
        self.architecture = self._get_architecture()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "bli_thread_get_num_threads", lambda: None)
        num_threads = get_func()
        # by default BLIS is single-threaded and get_num_threads
        # returns -1. We map it to 1 for consistency with other libraries.
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self.dynlib, "bli_thread_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    @_CDLL_CACHE.cache_method_on_dynlib
    def get_version(self):
        get_version_ = getattr(self.dynlib, "bli_info_get_version_str", None)
        if get_version_ is None:
            return None

        get_version_.restype = ctypes.c_char_p
        return get_version_().decode("utf-8")

    @_CDLL_CACHE.cache_method_on_dynlib
    def _get_threading_layer(self):
        """Return the threading layer of BLIS"""
        if getattr(self.dynlib, "bli_info_get_enable_openmp", lambda: False)():
            return "openmp"
        elif getattr(self.dynlib, "bli_info_get_enable_pthreads", lambda: False)():
            return "pthreads"
        return "disabled"

    @_CDLL_CACHE.cache_method_on_dynlib
    def _get_architecture(self):
        """Return the architecture detected by BLIS"""
        bli_arch_query_id = getattr(self.dynlib, "bli_arch_query_id", None)
        bli_arch_string = getattr(self.dynlib, "bli_arch_string", None)
        if bli_arch_query_id is None or bli_arch_string is None:
            return None

        # the true restype should be BLIS' arch_t (enum) but int should work
        # for us:
        bli_arch_query_id.restype = ctypes.c_int
        bli_arch_string.restype = ctypes.c_char_p
        return bli_arch_string(bli_arch_query_id()).decode("utf-8")


class FlexiBLASController(LibController):
    """Controller class for FlexiBLAS"""

    user_api = "blas"
    internal_api = "flexiblas"
    filename_prefixes = ("libflexiblas",)
    check_symbols = (
        "flexiblas_get_num_threads",
        "flexiblas_set_num_threads",
        "flexiblas_get_version",
        "flexiblas_list",
        "flexiblas_list_loaded",
        "flexiblas_current_backend",
    )

    @property
    def loaded_backends(self):
        return self._get_backend_list(loaded=True)

    @property
    def current_backend(self):
        return self._get_current_backend()

    def info(self, debugging_info: bool = False):
        """Return relevant info wrapped in a dict"""
        # We override the info method because the loaded and current backends
        # are dynamic properties
        exposed_attrs = super().info(debugging_info=debugging_info)
        exposed_attrs["loaded_backends"] = self.loaded_backends
        exposed_attrs["current_backend"] = self.current_backend

        return exposed_attrs

    def set_additional_attributes(self):
        self.available_backends = self._get_backend_list(loaded=False)

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "flexiblas_get_num_threads", lambda: None)
        num_threads = get_func()
        # by default BLIS is single-threaded and get_num_threads
        # returns -1. We map it to 1 for consistency with other libraries.
        return 1 if num_threads == -1 else num_threads

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self.dynlib, "flexiblas_set_num_threads", lambda num_threads: None
        )
        return set_func(num_threads)

    @_CDLL_CACHE.cache_method_on_dynlib
    def get_version(self):
        get_version_ = getattr(self.dynlib, "flexiblas_get_version", None)
        if get_version_ is None:
            return None

        major = ctypes.c_int()
        minor = ctypes.c_int()
        patch = ctypes.c_int()
        get_version_(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch))
        return f"{major.value}.{minor.value}.{patch.value}"

    def _get_backend_list(self, loaded=False):
        """Return the list of available backends for FlexiBLAS.

        If loaded is False, return the list of available backends from the FlexiBLAS
        configuration. If loaded is True, return the list of actually loaded backends.
        """
        func_name = f"flexiblas_list{'_loaded' if loaded else ''}"
        get_backend_list_ = getattr(self.dynlib, func_name, None)
        if get_backend_list_ is None:
            return None

        n_backends = get_backend_list_(None, 0, 0)

        backends = []
        for i in range(n_backends):
            backend_name = ctypes.create_string_buffer(1024)
            get_backend_list_(backend_name, 1024, i)
            if backend_name.value.decode("utf-8") != "__FALLBACK__":
                # We don't know when to expect __FALLBACK__ but it is not a real
                # backend and does not show up when running flexiblas list.
                backends.append(backend_name.value.decode("utf-8"))
        return backends

    def _get_current_backend(self):
        """Return the backend of FlexiBLAS"""
        get_backend_ = getattr(self.dynlib, "flexiblas_current_backend", None)
        if get_backend_ is None:
            return None

        backend = ctypes.create_string_buffer(1024)
        get_backend_(backend, ctypes.sizeof(backend))
        return backend.value.decode("utf-8")

    def switch_backend(self, backend):
        """Switch the backend of FlexiBLAS

        Parameters
        ----------
        backend : str
            The name or the path to the shared library of the backend to switch to. If
            the backend is not already loaded, it will be loaded first.
        """
        if backend not in self.loaded_backends:
            if backend in self.available_backends:
                load_func = getattr(self.dynlib, "flexiblas_load_backend", lambda _: -1)
            else:  # assume backend is a path to a shared library
                load_func = getattr(
                    self.dynlib, "flexiblas_load_backend_library", lambda _: -1
                )
            res = load_func(str(backend).encode("utf-8"))
            if res == -1:
                raise RuntimeError(
                    f"Failed to load backend {backend!r}. It must either be the name of"
                    " a backend available in the FlexiBLAS configuration "
                    f"{self.available_backends} or the path to a valid shared library."
                )

            # Invalidate the global cached controller:
            global _CACHED_CONTROLLER
            with _CACHE_LOCK:
                _CACHED_CONTROLLER = None

            # Trigger a new search of loaded shared libraries since loading a new
            # backend caused a dlopen.
            self.parent._load_libraries()

        switch_func = getattr(self.dynlib, "flexiblas_switch", lambda _: -1)
        idx = self.loaded_backends.index(backend)
        res = switch_func(idx)
        if res == -1:
            raise RuntimeError(f"Failed to switch to backend {backend!r}.")


class MKLController(LibController):
    """Controller class for MKL"""

    user_api = "blas"
    internal_api = "mkl"
    filename_prefixes = (
        "libmkl_rt",
        "mkl_rt",
        "libblas",  # legacy conda-forge Windows shim, see _make_controller_from_path
    )
    check_symbols = (
        "MKL_Get_Max_Threads",
        "MKL_Set_Num_Threads_Local",
        "MKL_Get_Version_String",
        "MKL_Set_Threading_Layer",
    )

    def set_additional_attributes(self):
        self.threading_layer = self._get_threading_layer()

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "MKL_Get_Max_Threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(
            self.dynlib, "MKL_Set_Num_Threads_Local", lambda num_threads: None
        )
        return set_func(num_threads)

    @_CDLL_CACHE.cache_method_on_dynlib
    def get_version(self):
        if not hasattr(self.dynlib, "MKL_Get_Version_String"):
            return None

        res = ctypes.create_string_buffer(200)
        self.dynlib.MKL_Get_Version_String(res, 200)

        version = res.value.decode("utf-8")
        group = re.search(r"Version ([^ ]+) ", version)
        if group is not None:
            version = group.groups()[0]
        return version.strip()

    @_CDLL_CACHE.cache_method_on_dynlib
    def _get_threading_layer(self):
        """Return the threading layer of MKL"""
        # The function mkl_set_threading_layer returns the current threading
        # layer. Calling it with an invalid threading layer allows us to safely
        # get the threading layer
        set_threading_layer = getattr(
            self.dynlib, "MKL_Set_Threading_Layer", lambda layer: -1
        )
        layer_map = {
            0: "intel",
            1: "sequential",
            2: "pgi",
            3: "gnu",
            4: "tbb",
            -1: "not specified",
        }
        return layer_map[set_threading_layer(-1)]


class OpenMPController(LibController):
    """Controller class for OpenMP"""

    user_api = "openmp"
    internal_api = "openmp"
    filename_prefixes = ("libiomp", "libgomp", "libomp", "vcomp")
    check_symbols = (
        "omp_get_max_threads",
        "omp_get_num_threads",
    )

    def get_num_threads(self):
        get_func = getattr(self.dynlib, "omp_get_max_threads", lambda: None)
        return get_func()

    def set_num_threads(self, num_threads):
        set_func = getattr(self.dynlib, "omp_set_num_threads", lambda num_threads: None)
        return set_func(num_threads)

    def get_version(self):
        # There is no way to get the version number programmatically in OpenMP.
        return None


# Controllers for the libraries that we'll look for in the loaded libraries.
# Third party libraries can register their own controllers.
_ALL_CONTROLLERS = [
    OpenBLASController,
    BLISController,
    MKLController,
    OpenMPController,
    FlexiBLASController,
]

# Helpers for the doc and test names
_ALL_USER_APIS = list({lib.user_api for lib in _ALL_CONTROLLERS})
_ALL_INTERNAL_APIS = [lib.internal_api for lib in _ALL_CONTROLLERS]
_ALL_PREFIXES = list(
    {prefix for lib in _ALL_CONTROLLERS for prefix in lib.filename_prefixes}
)
_ALL_BLAS_LIBRARIES = [
    lib.internal_api for lib in _ALL_CONTROLLERS if lib.user_api == "blas"
]
_ALL_OPENMP_LIBRARIES = OpenMPController.filename_prefixes


def register(controller):
    """Register a new controller"""
    _ALL_CONTROLLERS.append(controller)
    _ALL_USER_APIS.append(controller.user_api)
    _ALL_INTERNAL_APIS.append(controller.internal_api)
    _ALL_PREFIXES.extend(controller.filename_prefixes)
    # Libraries listed before this call may match the new controller:
    _CDLL_CACHE._controller_cache.clear()
    reset_cached_controller()


def _format_docstring(*args, **kwargs):
    def decorator(o):
        if o.__doc__ is not None:
            o.__doc__ = o.__doc__.format(*args, **kwargs)
        return o

    return decorator


@lru_cache(maxsize=10000)
def _realpath(filepath):
    """Small caching wrapper around os.path.realpath to limit system calls"""
    return os.path.realpath(filepath)


@_format_docstring(USER_APIS=list(_ALL_USER_APIS), INTERNAL_APIS=_ALL_INTERNAL_APIS)
def threadpool_info(debugging_info: bool = False):
    """Return the maximal number of threads for each detected library.

    Return a list with all the supported libraries that have been found. Each
    library is represented by a dict with the following information:

      - "user_api" : user API. Possible values are {USER_APIS}.
      - "internal_api": internal API. Possible values are {INTERNAL_APIS}.
      - "prefix" : filename prefix of the specific implementation.
      - "filepath": path to the loaded library.
      - "version": version of the library (if available).
      - "num_threads": the current thread limit.

    In addition, each library may contain internal_api specific entries.

    Parameters
    ----------
    debugging_info : bool
        Include extra fields which require more intrusive actions to obtain,
        and which can't always reliably be determined.

        - "thread_limit_scope": When setting the number of threads, what is
          affected. Possible values are "process", "current_thread", "unknown".
    """
    return ThreadpoolController().info(debugging_info)


_missing = object()


class _ThreadpoolLimiter:
    """The guts of ThreadpoolController.limit

    Refer to the docstring of ThreadpoolController.limit for more details.

    It will only act on the library controllers held by the provided `controller`.
    Using the default constructor sets the limits right away such that it can be used as
    a callable. Setting the limits can be delayed by using the `wrap` class method such
    that it can be used as a decorator.
    """

    def __init__(self, controller, *, limits=None, user_api=None):
        self._controller = controller
        self._limits, self._user_api = self._check_params(limits, user_api)
        self._original_info = self._controller.info()
        self._shared_pool_filepaths: list[str] = []
        # A plain call never reaches __exit__, so this increment is what makes the claim
        # stick until clear_controlling_thread(). A `with` block decrements the same
        # increment on the way out.
        self._apply_limits_with_shared_pool_claims()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.restore_original_limits()
        finally:
            self._release_shared_pool_claims()

    @classmethod
    def wrap(cls, controller, *, limits=None, user_api=None):
        """Return an instance of this class that can be used as a decorator"""
        return _ThreadpoolLimiterDecorator(
            controller=controller, limits=limits, user_api=user_api
        )

    def restore_original_limits(self):
        """Set the limits back to their original values"""
        for lib_controller, original_info in zip(
            self._controller.lib_controllers, self._original_info
        ):
            lib_controller.set_num_threads(original_info["num_threads"])

    # Alias of `restore_original_limits` for backward compatibility
    unregister = restore_original_limits

    def get_original_num_threads(self):
        """Original num_threads from before calling threadpool_limits

        Return a dict `{user_api: num_threads}`.
        """
        num_threads = {}
        warning_apis = []

        for user_api in self._user_api:
            limits = [
                lib_info["num_threads"]
                for lib_info in self._original_info
                if lib_info["user_api"] == user_api
            ]
            limits = set(limits)
            n_limits = len(limits)

            if n_limits == 1:
                limit = limits.pop()
            elif n_limits == 0:
                limit = None
            else:
                limit = min(limits)
                warning_apis.append(user_api)

            num_threads[user_api] = limit

        if warning_apis:
            warnings.warn(
                "Multiple values possible for following user apis: "
                + ", ".join(warning_apis)
                + ". Returning the minimum."
            )

        return num_threads

    def _check_params(self, limits, user_api):
        """Suitable values for the _limits and _user_api attributes"""

        if isinstance(limits, str) and limits == "sequential_blas_under_openmp":
            (
                limits,
                user_api,
            ) = self._controller._get_params_for_sequential_blas_under_openmp().values()

        if limits is None or isinstance(limits, int):
            if user_api is None:
                user_api = _ALL_USER_APIS
            elif user_api in _ALL_USER_APIS:
                user_api = [user_api]
            else:
                raise ValueError(
                    f"user_api must be either in {_ALL_USER_APIS} or None. Got "
                    f"{user_api} instead."
                )

            if limits is not None:
                limits = {api: limits for api in user_api}
        else:
            if isinstance(limits, list):
                # This should be a list of dicts of library info, for
                # compatibility with the result from threadpool_info.
                result = {}
                for lib_info in limits:
                    # Use prefix key for backwards compatibility with older
                    # versions that didn't use the filepath as the key.
                    if (filepath := lib_info.get("filepath", _missing)) is not _missing:
                        key = filepath
                    else:
                        key = lib_info["prefix"]
                    result[key] = lib_info["num_threads"]
                limits = result
            elif isinstance(limits, ThreadpoolController):
                # To set the limits from the library controllers of a
                # ThreadpoolController object.
                limits = {
                    lib_controller.filepath: lib_controller.num_threads
                    for lib_controller in limits.lib_controllers
                }

            if not isinstance(limits, dict):
                raise TypeError(
                    "limits must either be an int, a list, a dict, or "
                    f"'sequential_blas_under_openmp'. Got {type(limits)} instead"
                )

            # With a dictionary, can set both specific limit for given
            # libraries and global limit for user_api. Fetch each separately.
            user_api = [api for api in limits if api in _ALL_USER_APIS]

        return limits, user_api

    def _num_threads_for(self, lib_controller):
        """Limit that applies to ``lib_controller``, or None to leave it alone.

        ``self._limits`` is a dict ``{key: num_threads}`` where key is either a
        filepath, a prefix, or a user_api. Libraries are matched in that order.
        """
        if self._limits is None:
            return None
        if lib_controller.filepath in self._limits:
            num_threads = self._limits[lib_controller.filepath]
        elif lib_controller.prefix in self._limits:
            num_threads = self._limits[lib_controller.prefix]
        elif lib_controller.user_api in self._limits:
            num_threads = self._limits[lib_controller.user_api]
        else:
            return None
        return num_threads

    def _acquire_shared_pool_claims(self):
        """Count a scope per library before setting, so the claim is not sticky.

        Only libraries this limiter would set, and that this thread can
        control, are counted. OpenMP-only limits and ``limits=None`` do not
        take a claim. Libraries are counted separately so a scope that limits
        one copy does not pin another. A finished holder does not block this
        scope: counting first keeps the replacement claim non-sticky.
        """
        self._shared_pool_filepaths = []
        targets = [
            lib
            for lib in self._controller.lib_controllers
            if self._num_threads_for(lib) is not None
        ]
        for lib in _iter_claimable_shared_pools(targets):
            SHARED_POOL_CONTROL_OWNERSHIP._increment(lib.filepath)
            self._shared_pool_filepaths.append(lib.filepath)

    def _release_shared_pool_claims(self):
        filepaths = self._shared_pool_filepaths
        self._shared_pool_filepaths = []
        for filepath in filepaths:
            SHARED_POOL_CONTROL_OWNERSHIP._decrement(filepath)

    def _drop_unheld_shared_pool_claims(self):
        """Drop counts for libraries another thread won the race for."""
        this = current_thread()
        kept = []
        for filepath in self._shared_pool_filepaths:
            with _SHARED_POOL_STATES_LOCK:
                state = _SHARED_POOL_STATES.get(filepath)
            held = False
            if state is not None:
                with state.lock:
                    held = state.controlling_thread is this
            if held:
                kept.append(filepath)
            else:
                SHARED_POOL_CONTROL_OWNERSHIP._decrement(filepath)
        self._shared_pool_filepaths = kept

    def _apply_limits_with_shared_pool_claims(self):
        self._acquire_shared_pool_claims()
        try:
            self._set_threadpool_limits()
        except Exception:
            self._release_shared_pool_claims()
            raise
        self._drop_unheld_shared_pool_claims()

    def _set_threadpool_limits(self):
        """Change the maximal number of threads in selected thread pools."""
        if self._limits is None:
            return

        for lib_controller in self._controller.lib_controllers:
            num_threads = self._num_threads_for(lib_controller)
            if num_threads is not None:
                lib_controller.set_num_threads(num_threads)


class _ThreadpoolLimiterDecorator(_ThreadpoolLimiter, ContextDecorator):
    """Same as _ThreadpoolLimiter but to be used as a decorator"""

    def __init__(self, controller, *, limits=None, user_api=None):
        self._limits, self._user_api = self._check_params(limits, user_api)
        self._controller = controller
        self._shared_pool_filepaths = []

    def __enter__(self):
        # Set the limits here and not in __init__ so they apply when the decorated
        # function is called, not when the decorator is created. Increment before
        # setting so this scope's claim is not sticky.
        self._original_info = self._controller.info()
        self._apply_limits_with_shared_pool_claims()
        return self


@_format_docstring(
    USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
    BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
    OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
)
class threadpool_limits(_ThreadpoolLimiter):
    """Change the maximal number of threads that can be used in thread pools.

    This object can be used either as a callable (the construction of this object
    limits the number of threads), as a context manager in a `with` block to
    automatically restore the original state of the controlled libraries when exiting
    the block, or as a decorator through its `wrap` method.

    Set the maximal number of threads that can be used in thread pools used in
    the supported libraries to `limit`. This function works for libraries that
    are already loaded in the interpreter and can be changed dynamically.

    This effect is thread-local whenever possible but this is not guaranteed
    and actual behavior can depend on the controlled libraries. Notably,
    OpenBLAS built with the "windows" or "pthreads" threading layers uses a
    process-wide thread pool instead of independently controlled thread pools
    per calling threads.

    Parameters
    ----------
    limits : int, dict, list, 'sequential_blas_under_openmp' or None (default=None)
        The maximal number of threads that can be used in thread pools

        - If int, sets the maximum number of threads to `limits` for each
          library selected by `user_api`.

        - If it is a dictionary `{{key: max_threads}}`, this function sets a
          custom maximum number of threads for each `key` which can be either a
          `filepath`, `prefix`, or `user_api` for a specific library,
          checked in that order.

        - If it is a list, it should be a list of dictionaries of the form
          `{{"filepath": "/.../", "num_threads": 12}}`. Older versions of
          threadpoolctl accepted a `"prefix"` instead of `"filepath"`,
          so for maximal compatibility set both:
          `[{{"filepath": "...", "prefix": "...", "num_threads": 7}}]`

        - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
          and `user_api` parameters for the specific use case of sequential BLAS
          calls within an OpenMP parallel region. The `user_api` parameter is
          ignored.

        - If None, this function does not do anything.

    user_api : {USER_APIS} or None (default=None)
        APIs of libraries to limit. Used only if `limits` is an int.

        - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

        - If "openmp", it will only limit OpenMP supported libraries
          ({OPENMP_LIBS}). Note that it can affect the number of threads used
          by the BLAS libraries if they rely on OpenMP.

        - If None, this function will apply to all supported libraries.
    """

    def __init__(self, limits=None, user_api=None):
        super().__init__(ThreadpoolController(), limits=limits, user_api=user_api)

    @classmethod
    def wrap(cls, limits=None, user_api=None):
        return super().wrap(ThreadpoolController(), limits=limits, user_api=user_api)


_INCOMPATIBLE_OPENMP_MESSAGE = """
Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
the same time. Both libraries are known to be incompatible and this
can cause random crashes or deadlocks on Linux when loaded in the
same Python program.
Using threadpoolctl may cause crashes or deadlocks. For more
information and possible workarounds, please see
    https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md
"""


class ThreadpoolController:
    """Collection of LibController objects for all loaded supported libraries

    Attributes
    ----------
    lib_controllers : list of `LibController` objects
        The list of library controllers of all loaded supported libraries.
    """

    # Cache for libc under POSIX and a few system libraries under Windows.
    # We use a class level cache instead of an instance level cache because
    # it's very unlikely that a shared library will be unloaded and reloaded
    # during the lifetime of a program.
    _system_libraries = {}  # noqa: RUF012

    def __init__(self):
        self.lib_controllers = []
        self._load_libraries()
        self._warn_if_incompatible_openmp()

    @classmethod
    def _from_controllers(cls, lib_controllers):
        new_controller = cls.__new__(cls)
        new_controller.lib_controllers = lib_controllers
        return new_controller

    def info(self, debugging_info: bool = False):
        """Return lib_controllers info as a list of dicts.

        Parameters
        ----------
        debugging_info : bool
            Include extra fields which require more intrusive actions to
            obtain, and which can't always reliably be determined.
        """
        return [
            lib_controller.info(debugging_info=debugging_info)
            for lib_controller in self.lib_controllers
        ]

    def select(self, **kwargs):
        """Return a ThreadpoolController containing a subset of its current
        library controllers

        It will select all libraries matching at least one pair (key, value) from kwargs
        where key is an entry of the library info dict (like "user_api", "internal_api",
        "prefix", ...) and value is the value or a list of acceptable values for that
        entry.

        For instance, `ThreadpoolController().select(internal_api=["blis", "openblas"])`
        will select all library controllers whose internal_api is either "blis" or
        "openblas".
        """
        for key, vals in kwargs.items():
            kwargs[key] = [vals] if not isinstance(vals, list) else vals

        lib_controllers = [
            lib_controller
            for lib_controller in self.lib_controllers
            if any(
                getattr(lib_controller, key, None) in vals
                for key, vals in kwargs.items()
            )
        ]

        return ThreadpoolController._from_controllers(lib_controllers)

    def _get_params_for_sequential_blas_under_openmp(self):
        """Return appropriate params to use for a sequential BLAS call in an OpenMP loop

        This function takes into account the unexpected behavior of OpenBLAS with the
        OpenMP threading layer.
        """
        # select() matches any of its kwargs: chain two calls to require both.
        if (
            self.select(internal_api="openblas")
            .select(threading_layer="openmp")
            .lib_controllers
        ):
            # Don't change anything: OpenBLAS is already sequential by default when it
            # is called from an OpenMP parallel region, and limiting it would also
            # limit the OpenMP parallel regions of the calling thread.
            return {"limits": None, "user_api": None}
        return {"limits": 1, "user_api": "blas"}

    @_format_docstring(
        USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def limit(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used either as a callable (the
        construction of this object limits the number of threads) or as a context
        manager, in a `with` block to automatically restore the original state of the
        controlled libraries when exiting the block.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        This effect is thread-local whenever possible but this is not guaranteed
        and actual behavior can depend on the controlled libraries. Notably,
        OpenBLAS built with the "windows" or "pthreads" threading layers uses a
        process-wide thread pool instead of independently controlled thread pools
        per calling threads.

        Parameters
        ----------
        limits : int, dict, list, 'sequential_blas_under_openmp' or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `filepath`, `prefix`, or `user_api` for a specific library,
              checked in that order.

            - If it is a list, it should be a list of dictionaries of the form
              `{{"filepath": "/.../", "num_threads": 12}}`. Older versions of
              threadpoolctl accepted a `"prefix"` instead of `"filepath"`,
              so for maximal compatibility set both:
              `[{{"filepath": "...", "prefix": "...", "num_threads": 7}}]`

            - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
              and `user_api` parameters for the specific use case of sequential BLAS
              calls within an OpenMP parallel region. The `user_api` parameter is
              ignored.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter(self, limits=limits, user_api=user_api)

    @_format_docstring(
        USER_APIS=", ".join(f'"{api}"' for api in _ALL_USER_APIS),
        BLAS_LIBS=", ".join(_ALL_BLAS_LIBRARIES),
        OPENMP_LIBS=", ".join(_ALL_OPENMP_LIBRARIES),
    )
    def wrap(self, *, limits=None, user_api=None):
        """Change the maximal number of threads that can be used in thread pools.

        This function returns an object that can be used as a decorator.

        Set the maximal number of threads that can be used in thread pools used in
        the supported libraries to `limits`. This function works for libraries that
        are already loaded in the interpreter and can be changed dynamically.

        Parameters
        ----------
        limits : int, dict, list, 'sequential_blas_under_openmp', or None (default=None)
            The maximal number of threads that can be used in thread pools

            - If int, sets the maximum number of threads to `limits` for each
              library selected by `user_api`.

            - If it is a dictionary `{{key: max_threads}}`, this function sets a
              custom maximum number of threads for each `key` which can be either a
              `filepath`, `prefix`, or `user_api` for a specific library,
              checked in that order.

            - If it is a list, it should be a list of dictionaries of the form
              `{{"filepath": "/.../", "num_threads": 12}}`. Older versions of
              threadpoolctl accepted a `"prefix"` instead of `"filepath"`,
              so for maximal compatibility set both:
              `[{{"filepath": "...", "prefix": "...", "num_threads": 7}}]`

            - If 'sequential_blas_under_openmp', it will chose the appropriate `limits`
              and `user_api` parameters for the specific use case of sequential BLAS
              calls within an OpenMP parallel region. The `user_api` parameter is
              ignored.

            - If None, this function does not do anything.

        user_api : {USER_APIS} or None (default=None)
            APIs of libraries to limit. Used only if `limits` is an int.

            - If "blas", it will only limit BLAS supported libraries ({BLAS_LIBS}).

            - If "openmp", it will only limit OpenMP supported libraries
              ({OPENMP_LIBS}). Note that it can affect the number of threads used
              by the BLAS libraries if they rely on OpenMP.

            - If None, this function will apply to all supported libraries.
        """
        return _ThreadpoolLimiter.wrap(self, limits=limits, user_api=user_api)

    def __len__(self):
        return len(self.lib_controllers)

    def _load_libraries(self):
        """Loop through loaded shared libraries and store the supported ones"""
        # On glibc 2.39 and earlier, dl_iterate_phdr has an internal lock, and
        # that plus calling back into Python and the need to (re)acquire the
        # GIL can result in deadlocks.
        #
        # On glibc 2.40 and later, dl has a read-write lock, and
        # dl_iterate_phdr should only use the read version since it's not
        # modifying anything. So you no longer get deadlocks purely from
        # dl_iterate_phdr. You can still deadlock with dlopen() though.
        #
        # To avoid that deadlock, listing shared libraries can use a
        # Linux-specific mechanism that doesn't have these issues
        # (/proc/self/maps). Since it's the Linux kernel, musl works fine too.
        #
        # The downside is this mechanism is slower, so avoid it when safe
        # alternatives are available. In particular, if dllist() is available
        # (3.14+), written in C so no risk of GC part way (3.15+), and there is
        # no GIL, no need to use /proc, since dllist() is faster. It's possible
        # dllist() might work in GIL builds too but see
        # https://github.com/python/cpython/issues/157573. So we should
        # consider enabling it on GIL Python too once threadpoolctl supports
        # 3.15.
        if _HAS_PROCFS and not (
            sys.version_info[:2] >= (3, 15)
            and not getattr(sys, "_is_gil_enabled", lambda: True)()
        ):
            self._find_libraries_with_linux()
        elif dllist is not None:
            # On Python 3.14+, this functionality is built-in. Once Python 3.13
            # is no longer supported by threadpoolctl, most of the equivalent
            # threadpoolctl implementations can be removed.
            self._find_libraries_with_python(dllist)
        elif sys.platform == "darwin":
            self._find_libraries_with_dyld()
        elif sys.platform == "win32":
            self._find_libraries_on_windows()
        elif "pyodide" in sys.modules:
            self._find_libraries_pyodide()
        else:
            # Non-Linux Unix platforms.
            self._find_libraries_with_dl_iterate_phdr()

    _PATH_RE = re.compile(rb" (/[^\n]+\.so[^\n^/]*)\n", re.MULTILINE)

    def _find_libraries_with_linux(self):
        """Loop through loaded libraries and return binders on supported ones

        Uses a Linux-specific mechanism:
        https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html
        """
        with open("/proc/self/maps", "rb") as f:
            maps = f.read()
        filepaths = set(self._PATH_RE.findall(maps))
        for filepath in filepaths:
            filepath = filepath.decode("utf-8")
            self._make_controller_from_path(filepath)

    def _find_libraries_with_python(self, dllist):
        """Loop through loaded libraries and return binders on supported ones

        Uses Python's built-in support for this functionality.
        """
        try:
            filepaths = dllist()
        except OSError as exc:
            # No Toolhelp fallback: if this fires in the wild, collect a
            # minimal reproducer and report it upstream so CPython's dllist
            # can be hardened against concurrent module list changes.
            warnings.warn(
                "ctypes.util.dllist failed to list loaded libraries "
                f"({exc!r}). Native thread pools will not be inspected for "
                "this ThreadpoolController. Please report a minimal "
                "reproducer at https://github.com/joblib/threadpoolctl/issues "
                "so it can be investigated and possibly be reported upstream to "
                "CPython.",
                RuntimeWarning,
            )
            return
        if filepaths and filepaths[0] in ("", sys.executable):
            filepaths = filepaths[1:]
        for filepath in filepaths:
            self._make_controller_from_path(filepath)

    def _find_libraries_with_dl_iterate_phdr(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on POSIX system only.
        This code is adapted from code by Intel developer @anton-malakhov
        available at https://github.com/IntelPython/smp

        Copyright (c) 2017, Intel Corporation published under the BSD 3-Clause
        license
        """
        libc = self._get_libc()
        if not hasattr(libc, "dl_iterate_phdr"):  # pragma: no cover
            warnings.warn(
                "Could not find dl_iterate_phdr in the C standard library.",
                RuntimeWarning,
            )
            return []

        filepaths = []

        # Callback function for `dl_iterate_phdr` which is called for every
        # library loaded in the current process until it returns 1. To minimize
        # the potential for deadlocks (see #228), this code should not do
        # anything that might result in reentrancy into the library, the dl
        # system, or anything else.
        def match_library_callback(info, size, data):
            # Get the path of the current library
            filepath = info.contents.dlpi_name
            if filepath:
                filepaths.append(filepath)
            return 0

        c_func_signature = ctypes.CFUNCTYPE(
            ctypes.c_int,  # Return type
            ctypes.POINTER(_dl_phdr_info),
            ctypes.c_size_t,
            ctypes.c_char_p,
        )
        c_match_library_callback = c_func_signature(match_library_callback)

        data = ctypes.c_char_p(b"")
        libc.dl_iterate_phdr(c_match_library_callback, data)

        # Now that a list of filepaths is available, load the respective
        # libraries:
        for filepath in filepaths:
            # Store the library controller if it is supported and selected
            self._make_controller_from_path(filepath.decode("utf-8"))

    def _find_libraries_with_dyld(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on OSX system only
        """
        libc = self._get_libc()
        if not hasattr(libc, "_dyld_image_count"):  # pragma: no cover
            warnings.warn(
                "Could not find _dyld_image_count in the C standard library.",
                RuntimeWarning,
            )
            return []

        n_dyld = libc._dyld_image_count()
        libc._dyld_get_image_name.restype = ctypes.c_char_p

        for i in range(n_dyld):
            filepath = ctypes.string_at(libc._dyld_get_image_name(i))
            filepath = filepath.decode("utf-8")

            # Store the library controller if it is supported and selected
            self._make_controller_from_path(filepath)

    def _find_libraries_on_windows(self):
        """Loop through loaded libraries and return binders on supported ones

        This function is expected to work on windows system only.

        Used when ``ctypes.util.dllist`` is unavailable (Python < 3.14).
        Module discovery uses a snapshot-first strategy:
        ``CreateToolhelp32Snapshot`` provides an atomic list of loaded
        modules, which is more robust than ``EnumProcessModulesEx`` under
        concurrent DLL load/unload. Paths that fit in
        ``MODULEENTRY32W.szExePath`` (shorter than ``MAX_PATH``) are used as-is.
        Truncated or empty snapshot paths are resolved with
        ``GetModuleFileNameW`` and, when needed, ``GetModuleFileNameExW`` with a
        larger buffer. If snapshot creation fails, enumeration falls back to
        ``EnumProcessModulesEx``.
        """
        from ctypes.wintypes import MAX_PATH

        ps_api = self._get_windll("Psapi")
        kernel_32 = self._get_windll("kernel32")
        self._setup_windows_module_apis(ps_api, kernel_32)

        h_process = kernel_32.GetCurrentProcess()

        # Allocate a buffer for long path names; see _WINDOWS_MAX_LIBRARY_PATH_LENGTH.
        max_path = _WINDOWS_MAX_LIBRARY_PATH_LENGTH
        path_buf = ctypes.create_unicode_buffer(max_path)

        try:
            modules = self._snapshot_loaded_modules(kernel_32)
        except OSError:
            modules = None

        if modules is not None:
            for h_module, snapshot_path in modules:
                if snapshot_path and len(snapshot_path) < MAX_PATH - 1:
                    filepath = snapshot_path
                else:
                    filepath = self._resolve_module_filepath(
                        ps_api,
                        kernel_32,
                        h_process,
                        h_module,
                        max_path=max_path,
                        path_buf=path_buf,
                    )
                if filepath is not None:
                    self._make_controller_from_path(filepath)
        else:
            self._find_libraries_with_enum_process_modules_ex(
                ps_api, kernel_32, h_process, max_path, path_buf
            )

    @classmethod
    def _setup_windows_module_apis(cls, ps_api, kernel_32):
        """Set ctypes signatures for Windows module enumeration APIs."""
        from ctypes.wintypes import BOOL, DWORD, HANDLE, HMODULE

        if getattr(cls, "_windows_module_apis_configured", False):
            return

        kernel_32.GetCurrentProcess.restype = HANDLE
        kernel_32.CreateToolhelp32Snapshot.argtypes = [DWORD, DWORD]
        kernel_32.CreateToolhelp32Snapshot.restype = HANDLE
        kernel_32.Module32FirstW.argtypes = [HANDLE, ctypes.c_void_p]
        kernel_32.Module32FirstW.restype = BOOL
        kernel_32.Module32NextW.argtypes = [HANDLE, ctypes.c_void_p]
        kernel_32.Module32NextW.restype = BOOL
        kernel_32.GetModuleFileNameW.argtypes = [HMODULE, ctypes.c_wchar_p, DWORD]
        kernel_32.GetModuleFileNameW.restype = DWORD
        kernel_32.CloseHandle.argtypes = [HANDLE]
        kernel_32.CloseHandle.restype = BOOL

        ps_api.EnumProcessModulesEx.argtypes = [
            HANDLE,
            ctypes.POINTER(HMODULE),
            DWORD,
            ctypes.POINTER(DWORD),
            DWORD,
        ]
        ps_api.EnumProcessModulesEx.restype = BOOL
        ps_api.GetModuleFileNameExW.argtypes = [
            HANDLE,
            HMODULE,
            ctypes.c_wchar_p,
            DWORD,
        ]
        ps_api.GetModuleFileNameExW.restype = DWORD

        cls._windows_module_apis_configured = True

    def _snapshot_loaded_modules(self, kernel_32):
        """Return loaded modules as (hModule, snapshot_path) pairs.

        Uses CreateToolhelp32Snapshot for an atomic view of loaded modules, which
        is more robust than EnumProcessModulesEx when DLLs are loaded or unloaded
        concurrently. ``ERROR_BAD_LENGTH`` is retried a bounded number of times
        (the documented transient race when the module list changes mid-snapshot)
        and then raised as ``OSError`` so the caller can fall back.
        """
        from ctypes.wintypes import DWORD, HANDLE, MAX_PATH

        class MODULEENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", DWORD),
                ("th32ModuleID", DWORD),
                ("th32ProcessID", DWORD),
                ("GlblcntUsage", DWORD),
                ("ProccntUsage", DWORD),
                ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
                ("modBaseSize", DWORD),
                ("hModule", HANDLE),
                ("szModule", ctypes.c_wchar * 256),
                ("szExePath", ctypes.c_wchar * MAX_PATH),
            ]

        TH32CS_SNAPMODULE = 0x00000008
        TH32CS_SNAPMODULE32 = 0x00000010
        ERROR_BAD_LENGTH = 0x0018
        ERROR_NO_MORE_FILES = 0x0012
        INVALID_HANDLE_VALUE = HANDLE(-1).value
        max_snapshot_retries = 16

        for _ in range(max_snapshot_retries):
            snap_handle = kernel_32.CreateToolhelp32Snapshot(
                TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, os.getpid()
            )
            if snap_handle != INVALID_HANDLE_VALUE:
                break
            err = ctypes.get_last_error()
            if err == ERROR_BAD_LENGTH:
                continue
            msg = ctypes.FormatError(err).strip()
            raise OSError(f"CreateToolhelp32Snapshot failed: {msg}")
        else:
            msg = ctypes.FormatError(ERROR_BAD_LENGTH).strip()
            raise OSError(f"CreateToolhelp32Snapshot failed: {msg}")

        modules = []
        try:
            lib_entry = MODULEENTRY32W()
            lib_entry.dwSize = ctypes.sizeof(MODULEENTRY32W)
            if not kernel_32.Module32FirstW(
                snap_handle, ctypes.byref(lib_entry)
            ):  # pragma: no cover
                err = ctypes.get_last_error()
                msg = ctypes.FormatError(err).strip()
                raise OSError(f"Module32FirstW failed: {msg}")

            while True:
                modules.append((lib_entry.hModule, lib_entry.szExePath))
                if not kernel_32.Module32NextW(snap_handle, ctypes.byref(lib_entry)):
                    err = ctypes.get_last_error()
                    if err != ERROR_NO_MORE_FILES:  # pragma: no cover
                        msg = ctypes.FormatError(err).strip()
                        raise OSError(f"Module32NextW failed: {msg}")
                    break
        finally:
            kernel_32.CloseHandle(snap_handle)

        return modules

    def _resolve_module_filepath(
        self,
        ps_api,
        kernel_32,
        h_process,
        h_module,
        max_path,
        path_buf,
    ):
        """Return the full path for a module, or None if it should be skipped."""
        from ctypes.wintypes import MAX_PATH

        n_size = kernel_32.GetModuleFileNameW(h_module, path_buf, MAX_PATH)
        if n_size and n_size < MAX_PATH - 1:
            return path_buf.value

        n_size = ps_api.GetModuleFileNameExW(h_process, h_module, path_buf, max_path)
        if n_size and n_size < max_path - 1:
            return path_buf.value
        if n_size:  # pragma: no cover
            filepath = path_buf.value
            warnings.warn(
                "Could not get the full path of a dynamic library (path too "
                "long). This library will be ignored and threadpoolctl might "
                "not be able to control or display information about all "
                f"loaded libraries. Here's the truncated path: {filepath!r}",
                RuntimeWarning,
            )
            return None

        err = ctypes.get_last_error()
        msg = ctypes.FormatError(err).strip()
        warnings.warn(
            "Could not get the path of a dynamic library "
            "(GetModuleFileNameW and GetModuleFileNameExW failed). "
            "This library will be ignored and threadpoolctl might not be "
            "able to control or display information about all loaded "
            f"libraries. {msg}",
            RuntimeWarning,
        )
        return None

    def _find_libraries_with_enum_process_modules_ex(
        self, ps_api, kernel_32, h_process, max_path, path_buf
    ):
        """Fallback Windows module enumeration using EnumProcessModulesEx.

        This code is adapted from code by Philipp Hagemeister @phihag available
        at https://stackoverflow.com/questions/17474574
        """
        from ctypes.wintypes import DWORD, HMODULE

        LIST_LIBRARIES_ALL = 0x03

        buf_count = 256
        needed = DWORD()
        # Grow the buffer until it becomes large enough to hold all the
        # module headers
        while True:
            buf = (HMODULE * buf_count)()
            buf_size = ctypes.sizeof(buf)
            if not ps_api.EnumProcessModulesEx(
                h_process,
                buf,
                buf_size,
                ctypes.byref(needed),
                LIST_LIBRARIES_ALL,
            ):
                raise OSError("EnumProcessModulesEx failed")
            if buf_size >= needed.value:
                break
            buf_count = needed.value // (buf_size // buf_count)

        count = needed.value // (buf_size // buf_count)
        for h_module in map(HMODULE, buf[:count]):
            filepath = self._resolve_module_filepath(
                ps_api,
                kernel_32,
                h_process,
                h_module,
                max_path=max_path,
                path_buf=path_buf,
            )
            if filepath is not None:
                self._make_controller_from_path(filepath)

    def _find_libraries_pyodide(self):
        """Pyodide specific implementation for finding loaded libraries.

        Adapted from suggestion in https://github.com/joblib/threadpoolctl/pull/169#issuecomment-1946696449.

        One day, we may have a simpler solution. libc dl_iterate_phdr needs to
        be implemented in Emscripten and exposed in Pyodide, see
        https://github.com/emscripten-core/emscripten/issues/21354 for more
        details.
        """
        try:
            from pyodide_js._module import LDSO
        except ImportError:
            warnings.warn(
                "Unable to import LDSO from pyodide_js._module. This should never "
                "happen."
            )
            return

        if hasattr(LDSO.loadedLibsByName, "as_py_json"):  # Pyodide >= 0.29
            libs_iter = LDSO.loadedLibsByName.as_py_json()
        else:
            libs_iter = LDSO.loadedLibsByName.as_object_map()  # Pyodide < 0.29

        for filepath in libs_iter:
            # Some libraries are duplicated by Pyodide and do not exist in the
            # filesystem, so we first check for the existence of the file. For
            # more details, see
            # https://github.com/joblib/threadpoolctl/pull/169#issuecomment-1947946728
            if os.path.exists(filepath):
                self._make_controller_from_path(filepath)

    def _make_controller_from_path(self, filepath):
        """Store a library controller if it is supported and selected"""
        lib_controller = _CDLL_CACHE.create_controller(filepath, self)
        if lib_controller is not None:
            if lib_controller.filepath in (
                lib.filepath for lib in self.lib_controllers
            ):
                # We already have a controller for this library.
                return

            self.lib_controllers.append(lib_controller)

    def _warn_if_incompatible_openmp(self):
        """Raise a warning if llvm-OpenMP and intel-OpenMP are both loaded"""
        if sys.platform != "linux":
            # The incompatibility between libomp and libiomp is known to cause
            # crashes on Linux. On other platforms, conda-forge may expose both
            # libraries without the same runtime conflict (for instance when
            # libiomp forwards to libomp on Windows).
            return

        prefixes = [lib_controller.prefix for lib_controller in self.lib_controllers]
        if "libomp" in prefixes and "libiomp" in prefixes:
            warnings.warn(_INCOMPATIBLE_OPENMP_MESSAGE, RuntimeWarning)

    @classmethod
    def _get_libc(cls):
        """Load the lib-C for unix systems."""
        libc = cls._system_libraries.get("libc")
        if libc is None:
            # dlopen(NULL) rather than ctypes.util.find_library("c"). Importing
            # ctypes.util on CPython 3.14 Linux creates a process-lifetime
            # CFUNCTYPE callback that is not fork-safe with some libffi builds
            # (issue #225). If libc is statically linked or Python is linked
            # against musl, the main program still exports the libc symbols we
            # need. If it does not, we check for those symbols later anyway.
            libc = ctypes.CDLL(None, mode=_RTLD_NOLOAD)
            cls._system_libraries["libc"] = libc
        return libc

    @classmethod
    def _get_windll(cls, dll_name):
        """Load a windows DLL"""
        dll = cls._system_libraries.get(dll_name)
        if dll is None:
            dll = ctypes.WinDLL(f"{dll_name}.dll", use_last_error=True)
            cls._system_libraries[dll_name] = dll
        return dll


def _main():
    """Commandline interface to display thread-pool information and exit."""
    import argparse
    import importlib
    import json
    import sys

    parser = argparse.ArgumentParser(
        usage="python -m threadpoolctl -i numpy scipy.linalg xgboost",
        description="Display thread-pool information and exit.",
    )
    parser.add_argument(
        "-i",
        "--import",
        dest="modules",
        nargs="*",
        default=(),
        help="Python modules to import before introspecting thread-pools.",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="a Python statement to execute before introspecting thread-pools.",
    )

    options = parser.parse_args(sys.argv[1:])
    for module in options.modules:
        try:
            importlib.import_module(module, package=None)
        except ImportError:
            print("WARNING: could not import", module, file=sys.stderr)

    if options.command:
        exec(options.command)  # noqa: S102

    print(json.dumps(threadpool_info(debugging_info=True), indent=2))


# A cached instance of ThreadpoolController, created by
# get_cached_controller():
_CACHED_CONTROLLER = None
# Cached number of modules in sys.modules, used by get_cached_controller()
_CACHED_SYS_MODULES_LEN = 0
# The lock controlling access to the above:
_CACHE_LOCK = Lock()


def get_cached_controller() -> ThreadpoolController:
    """Return a cached ``ThreadpoolController``, as a speed optimization.

    The cached version is regenerated if the number of imported of modules has
    changed, since this indicates the possibility that a new controllable
    library has been loaded.  Invalidation also happens if Flexiblas is in use
    and has changed its backend.

    It is possible that somehow a controllable library gets loaded without
    importing a Python module, e.g. using ``dlopen()`` on Linux, a ctypes
    ``CDLL``, and the like.  This is rare, however.  Outside of these
    situations, using ``get_cached_controller()`` will be much faster than
    creating a new ``ThreadpoolController`` instance.

    The cached controller can be explicitly invalidated by calling the
    `reset_cached_controller()` module level function.
    """
    global _CACHED_CONTROLLER, _CACHED_SYS_MODULES_LEN
    # Fast path with no lock: snapshot before the `None` check as
    # `reset_cached_controller()` and `FlexiBLASController.switch_backend()`
    # can concurrently set the global to `None`. A stale controller instance is
    # acceptable but returning `None` would break the caller.
    controller = _CACHED_CONTROLLER
    if _CACHED_SYS_MODULES_LEN == len(sys.modules) and controller is not None:
        return controller
    with _CACHE_LOCK:
        # Do this once, so there's no race condition with imports in other
        # threads:
        num_modules = len(sys.modules)
        if _CACHED_CONTROLLER is None or num_modules != _CACHED_SYS_MODULES_LEN:
            _CACHED_CONTROLLER = ThreadpoolController()
            _CACHED_SYS_MODULES_LEN = num_modules
        return _CACHED_CONTROLLER


def reset_cached_controller() -> None:
    """Clear the cached ``ThreadpoolController``, if any."""
    global _CACHED_CONTROLLER, _CACHED_SYS_MODULES_LEN
    with _CACHE_LOCK:
        _CACHED_CONTROLLER = None
        _CACHED_SYS_MODULES_LEN = 0


if __name__ == "__main__":
    _main()
