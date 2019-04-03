#!/bin/bash

set -e

if [[ "$PACKAGER" == "conda" ]]; then
    source activate $VIRTUALENV
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source $VIRTUALENV/bin/activate
fi

python -c "import numpy; from tests._openmp_test_helper_outer import *; from threadpoolctl import threadpool_info; from pprint import pprint; pprint(threadpool_info())"
set -x
pytest -vlx --junitxml=$JUNITXML --cov=threadpoolctl
set +x
