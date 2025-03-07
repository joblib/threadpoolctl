#!/bin/bash

set -xe

if [[ "$PACKAGER" == conda* ]] || [[ -z "$PACKAGER" ]]; then
    conda activate testenv
    conda list
elif [[ "$PACKAGER" == pip* ]]; then
    # we actually use conda to install the base environment:
    conda activate testenv
    pip list
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source testenv/bin/activate
    pip list
fi

export MKL_THREADING_LAYER=INTEL


# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg #tests._openmp_test_helper.openmp_helpers_inner

pytest -vlrxXs -W error -k "$TESTS" --junitxml=$JUNITXML --cov=threadpoolctl --cov-report xml
