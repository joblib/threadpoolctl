#!/bin/bash

set -e

if [[ "$PACKAGER" == conda* ]]; then
    source activate $VIRTUALENV
    conda list
elif [[ "$PACKAGER" == "pip" ]]; then
    # we actually use conda to install the base environment:
    source activate $VIRTUALENV
    pip list
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
    pip list
fi

set -x

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper.openmp_helpers_inner

pytest -vlrxXs -W error -k "$TESTS" --junitxml=$JUNITXML --cov=threadpoolctl
