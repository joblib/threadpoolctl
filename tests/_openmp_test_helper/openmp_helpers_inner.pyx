cimport openmp
from cython.parallel import prange
from pprint import pprint
from threadpoolctl import threadpool_info


def check_openmp_num_threads(int n):
    """Run a short parallel section with OpenMP

    Return the number of threads that where effectively used by the
    OpenMP runtime.
    """
    cdef int num_threads = -1

    with nogil:
        num_threads = inner_openmp_loop(n, 0)
    return num_threads


def print_omp_debug_info(outerloop_idx, num_threads):
    msg = "[%d] " % outerloop_idx
    msg += ", ".join(m['prefix'] + ": %d" % m['num_threads']
                    for m in threadpool_info()
                    if m["user_api"] == "openmp")
    msg += ", inner loop omp_get_num_threads: %d" % num_threads
    print(msg, flush=True)


cdef int inner_openmp_loop(int n, int outerloop_idx) nogil:
    """Run a short parallel section with OpenMP

    Return the number of threads that where effectively used by the
    OpenMP runtime.

    This function is expected to run without the GIL and can be called
    by an outer OpenMP / prange loop written in Cython in anoter file.
    """
    cdef long n_sum = 0
    cdef int i, num_threads = 0

    for i in prange(n):
        num_threads = openmp.omp_get_num_threads()
        n_sum += i
        with gil:
            print_omp_debug_info(outerloop_idx, num_threads)

    if n_sum != (n - 1) * n / 2:
        # error
        return -1

    return num_threads


def get_compiler():
    return CC_INNER_LOOP
