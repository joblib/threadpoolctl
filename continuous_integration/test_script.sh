#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x

python -c "from tests.test_threadpoolctl import test_nested_prange_blas; test_nested_prange_blas(1)"
python -c "from tests.test_threadpoolctl import test_nested_prange_blas; test_nested_prange_blas(2)"
python -c "from tests.test_threadpoolctl import test_nested_prange_blas; test_nested_prange_blas(4)"
python -c "from tests.test_threadpoolctl import test_nested_prange_blas; test_nested_prange_blas(None)"

pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl -k openmp_nesting
set +x
