cimport openmp
from cython.parallel import prange
from openmp_helpers_inner cimport inner_openmp_loop


def check_nested_openmp_loops(int n, nthreads=None):
    """Run a short parallel section with OpenMP with nested calls

    The inner OpenMP loop has not necessarily been built/linked with the
    same runtime OpenMP runtime.
    """
    cdef:
        int outer_num_threads = -1
        int inner_num_threads = -1
        int num_threads = nthreads or openmp.omp_get_max_threads()
        int i

    for i in prange(n, num_threads=num_threads, nogil=True):
        inner_num_threads = inner_openmp_loop(n)
        outer_num_threads = openmp.omp_get_num_threads()
        
    return outer_num_threads, inner_num_threads
