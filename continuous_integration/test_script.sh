#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl -k "limit_num_threads or prange_blas"
set +x
