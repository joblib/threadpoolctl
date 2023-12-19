cimport openmp
from cython.parallel import prange


def check_openmp_num_threads(int n):
    """Run a short parallel section with OpenMP

    Return the number of threads that where effectively used by the
    OpenMP runtime.
    """
    cdef int num_threads = -1

    with nogil:
        num_threads = inner_openmp_loop(n)
    return num_threads


cdef int inner_openmp_loop(int n) noexcept nogil:
    """Run a short parallel section with OpenMP

    Return the number of threads that where effectively used by the
    OpenMP runtime.

    This function is expected to run without the GIL and can be called
    by an outer OpenMP / prange loop written in Cython in another file.
    """
    cdef long n_sum = 0
    cdef int i, num_threads

    for i in prange(n):
        num_threads = openmp.omp_get_num_threads()
        n_sum += i

    if n_sum != (n - 1) * n / 2:
        # error
        return -1

    return num_threads
