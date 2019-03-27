#!/bin/bash

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    conda activate $VIRTUALENV
elif [[ "$DISTRIB" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

TEST_CMD="python -m pytest -vl --junitxml=$JUNITXML --cov=threadpoolctl"

set -x
$TEST_CMD
set +x
