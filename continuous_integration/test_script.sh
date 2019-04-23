#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

set -x
PYTHONPATH="." python continuous_integration/display_versions.py

pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl &
sleep 5
sudo gdb -p $! -ex "set confirm off" -ex "thread apply all bt" -ex q
fg

set +x
