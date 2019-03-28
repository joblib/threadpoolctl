#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
pytest -vlx --junitxml=$JUNITXML \
    --cov=threadpoolctl --cov-config=$COVERAGE_DATA
set +x
