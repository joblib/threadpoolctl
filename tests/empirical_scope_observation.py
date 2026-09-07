"""
Run this script to empirically determine if OpenMP and BLAS limiting API set
the size of a shared process-wide thread pool or a per-thread limit.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
from pprint import pprint
from time import time
from threading import Thread
from typing import Callable

import threadpoolctl
import psutil


def start_counting_threads() -> Callable[[], int]:
    max_num_threads = 0
    stop = False

    def in_thread():
        nonlocal max_num_threads, stop
        process = psutil.Process()
        while not stop:
            max_num_threads = max(max_num_threads, process.num_threads())

    thread = Thread(target=in_thread)
    thread.start()

    def finish():
        nonlocal stop, max_num_threads
        stop = True
        thread.join()
        return max_num_threads

    return finish


def set_limits():
    threadpoolctl.threadpool_limits(2)
    print("== Library info ==")
    pprint(threadpoolctl.threadpool_info())
    print()


def blas(num_python_threads):
    try:
        import numpy as np
    except ImportError:
        print("BLAS not available")
        return False

    set_limits()
    A = np.ones((1024, 1024))

    def blas_math(_):
        A.dot(A)

    with ThreadPoolExecutor(num_python_threads) as pool:
        t0 = time()
        while time() - t0 < 5:
            for _ in pool.map(blas_math, range(40)):
                pass

    return True


def openmp(num_python_threads):
    try:
        from tests._openmp_test_helper.openmp_helpers_inner import (
            check_openmp_num_threads,
        )
    except ImportError:
        print("OpenMP not available")
        return False

    set_limits()

    def run_openmp(_):
        check_openmp_num_threads(1000)

    with ThreadPoolExecutor(num_python_threads) as pool:
        t0 = time()
        while time() - t0 < 5:
            for _ in pool.map(run_openmp, range(40)):
                pass

    return True


def run(which: str) -> None:
    if which == "blas":
        func = blas
    else:
        func = openmp

    num_python_threads = 10 * cpu_count()
    count_threads = start_counting_threads()
    if not func(num_python_threads):
        count_threads()
        return
    max_num_threads = count_threads()

    # We're creating os.cpu_count() * 10 Python threads, and asking for 2
    # native threads. If number is close to number of Python threads, we'll
    # assume process-wide shared thread pool. If the number is close to double
    # the number of Python threads, we'll assume a thread pool per thread.
    if max_num_threads < num_python_threads * 1.25:
        scope = "process-wide shared thread pool"
    elif max_num_threads > num_python_threads * 1.75:
        scope = "thread pool per thread"
    else:
        scope = "not sure"

    print(f"== Observed behavior: {which} ==")
    print(
        f"10x{cpu_count()} Python threads each running a parallel "
        "workload with a 2 thread limit."
    )
    print(
        f"Maximum number of observed threads (includes Python and {which} threads):",
        max_num_threads,
    )
    print("Presumed API scope:", scope)
    print()


if __name__ == "__main__":
    if sys.argv[1] == "openmp":
        run("openmp")
    elif sys.argv[1] == "blas":
        run("blas")
    else:
        raise SystemExit("Bad argument")
