#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(1); test_nested_prange_blas(1)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(1); test_nested_prange_blas(2)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(1); test_nested_prange_blas(4)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(1); test_nested_prange_blas(None)"

python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(2); test_nested_prange_blas(1)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(2); test_nested_prange_blas(2)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(2); test_nested_prange_blas(4)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(2); test_nested_prange_blas(None)"

python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(4); test_nested_prange_blas(1)"
# python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(4); test_nested_prange_blas(2)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(4); test_nested_prange_blas(4)"
python -c "from tests.test_threadpoolctl import test_openmp_limit_num_threads, test_nested_prange_blas; test_openmp_limit_num_threads(4); test_nested_prange_blas(None)"

# pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl
set +x
