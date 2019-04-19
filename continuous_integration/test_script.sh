#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
pytest -vl --junitxml=$JUNITXML -k openmp_nesting
pytest -vl --junitxml=$JUNITXML -k prange
pytest -vl --junitxml=$JUNITXML -k "openmp_nesting or prange"
pytest -vl --junitxml=$JUNITXML

pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl -k "openmp_nesting or prange"
set +x
