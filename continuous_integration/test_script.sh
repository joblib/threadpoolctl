#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "pip" ]]; then
    # we actually use conda to install the base environment:
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
PYTHONPATH="." python continuous_integration/display_versions.py

pytest -vlrxXs --junitxml=$JUNITXML --cov=threadpoolctl
