#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    conda activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl
set +x
