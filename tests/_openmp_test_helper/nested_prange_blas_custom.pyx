cimport openmp
from cython.parallel import parallel, prange

import numpy as np

cdef extern from 'cblas.h' nogil:
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor=101
        CblasColMajor=102
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
    void dgemm 'cblas_dgemm' (
        CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB, int M, int N,
        int K, double alpha, double *A, int lda,
        double *B, int ldb, double beta, double *C, int ldc)

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
            dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            chunk_size, n, k, alpha, &A[i * chunk_size, 0], k,
            &B[0, 0], k, beta, &C[i * chunk_size, 0], n)

    return np.asarray(C), prange_num_threads, inner_info[0]
