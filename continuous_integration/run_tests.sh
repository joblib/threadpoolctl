#!/bin/bash

set -xe

if [[ "$PACKAGER" == conda* ]] || [[ -z "$PACKAGER" ]]; then
    source "$CONDA/etc/profile.d/conda.sh"
    conda activate testenv
    conda list
elif [[ "$PACKAGER" == pip* ]]; then
    # we actually use conda to install the base environment:
    source "$CONDA/etc/profile.d/conda.sh"
    conda activate testenv
    pip list
elif [[ "$PACKAGER" == "ubuntu" ]]; then
    source testenv/bin/activate
    pip list
fi

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper.openmp_helpers_inner

# Introspect API scope:
PYTHONPATH=. python tests/empirical_scope_observation.py blas
PYTHONPATH=. python tests/empirical_scope_observation.py openmp

pytest -vlrxXs -W error -k "$TESTS" --junitxml=test_result.xml --cov=threadpoolctl --cov-report xml
