cimport openmp
from cython.parallel import prange
from openmp_helpers_inner cimport inner_openmp_loop

import numpy as np
from scipy.linalg.cython_blas cimport dgemm

from threadpoolctl import threadpool_info


def check_nested_openmp_loops(int n):
    """Run a short parallel section with OpenMP with nested calls

    The inner OpenMP loop has not necessarily been built/linked with the
    same runtime OpenMP runtime.
    """
    cdef int outer_num_threads, inner_num_threads, i

    for i in prange(n, nogil=True):
        inner_num_threads = inner_openmp_loop(n)
        outer_num_threads = openmp.omp_get_num_threads()
        
    return outer_num_threads, inner_num_threads


def check_nested_prange_blas(double[:, ::1] A, double[:, ::1] B, int nthreads):
    """Run multithreaded BLAS calls within OpenMP parallel loop"""
    cdef:
        int m = A.shape[0]
        int n = B.shape[0]
        int k = A.shape[1]

        double[:, ::1] C = np.empty((m, n))
        int n_chunks = 100
        int chunk_size = A.shape[0] // n_chunks

        char* trans = 't'
        char* no_trans = 'n'
        double alpha = 1.0
        double beta = 0.0

        int i
        int prange_num_threads
    
    blas_num_threads = []

    for i in prange(n_chunks, num_threads=nthreads, nogil=True):
        dgemm(trans, no_trans, &n, &chunk_size, &k,
              &alpha, &B[0, 0], &k, &A[i * chunk_size, 0], &k,
              &beta, &C[i * chunk_size, 0], &n)
    
        prange_num_threads = openmp.omp_get_num_threads()

        if i == 0:
            with gil:
                infos = threadpool_info()
                for lib in infos:
                    if lib['user_api'] == 'blas':
                        blas_num_threads.append(lib['num_threads'])

    return np.asarray(C), prange_num_threads, blas_num_threads


def get_compiler():
    return CC_OUTER_LOOP
