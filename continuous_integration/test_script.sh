#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x

tests=(
    limits_by_prefix
    limits_by_api
    limits_function_with_side_effect
    limits_no_limit
    limits_manual_unregister
    limits_bad_input
    limit_num_threads
    openmp_nesting
    shipped_openblas
)
for tst in "${tests[@]}"; do
    pytest -vl --junitxml=$JUNITXML -k "$tst or prange_blas"
done

for tst in "${tests[@]}"; do
    pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl -k "$tst or prange_blas"
done

set +x
