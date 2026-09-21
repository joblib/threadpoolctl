# cython: freethreading_compatible = True

cimport openmp
from cython.parallel import parallel, prange

import numpy as np
from scipy.linalg.cython_blas cimport dgemm

from threadpoolctl import ThreadpoolController


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
        int *prange_num_threads_ptr = &prange_num_threads

    inner_info = [None]

    with nogil, parallel(num_threads=nthreads):
        if openmp.omp_get_thread_num() == 0:
            with gil:
                inner_info[0] = ThreadpoolController().info()

            prange_num_threads_ptr[0] = openmp.omp_get_num_threads()

        for i in prange(n_chunks):
            dgemm(trans, no_trans, &n, &chunk_size, &k,
                &alpha, &B[0, 0], &k, &A[i * chunk_size, 0], &k,
                &beta, &C[i * chunk_size, 0], &n)

    return np.asarray(C), prange_num_threads, inner_info[0]
