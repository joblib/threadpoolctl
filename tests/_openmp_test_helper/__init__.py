from .openmp_helpers_inner import check_openmp_num_threads
from .openmp_helpers_outer import check_nested_openmp_loops

try:
    from .nested_prange_blas import check_nested_prange_blas
except ImportError:
    # Can happen if numpy and scipy are missing.
    check_nested_prange_blas = None


__all__ = ["check_openmp_num_threads",
           "check_nested_openmp_loops",
           "check_nested_prange_blas"]
