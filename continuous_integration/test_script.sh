#!/bin/bash

set -e

if [[ "$PACKAGER" == conda* ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "pip" ]]; then
    # we actually use conda to install the base environment:
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper

pytest -vlrxXs -W error -k "$TESTS" --junitxml=$JUNITXML --cov=threadpoolctl
