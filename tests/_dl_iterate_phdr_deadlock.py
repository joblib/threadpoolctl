"""Try to trigger deadlock via dl_iterate_phdr."""

import ctypes
import os
import sys
from threading import Thread

from threadpoolctl import threadpool_limits


def create_controllers(done):
    for _ in range(100):
        # May use dl_iterate_phdr() on Linux:
        limiter = threadpool_limits()
        # dlopen() of some shared libraries like to be installed:
        loaded = False
        # Common libraries on Ubuntu:
        for name in [
            "libncurses.so.6",
            "libncursesw.so.6",
            "libgmp.so.10",
            "libssl.so.3",
            "libcrypt.so.1",
        ]:
            try:
                dll = ctypes.CDLL(name)
                del dll
                loaded = True
            except OSError:
                pass
        if not loaded:
            # Couldn't do an dlopen()s.
            os._exit(7)

        # Imports, which also do dlopen():
        try:
            import numpy  # also gets us BLAS
        except ImportError:
            pass
        import _pickle

        try:
            import tests._openmp_test_helper.nested_prange_blas
        except ImportError:
            pass

        del limiter

    done.append(True)


def main():
    threads = []
    done = []

    for _ in range(os.cpu_count() * 4):
        t = Thread(target=create_controllers, args=(done,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if len(done) != os.cpu_count() * 4:
        sys.exit(1)

    # Special success exist code:
    sys.exit(17)


if __name__ == "__main__":
    main()
