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

if [[ "$PYTHON_FREETHREADED" == "true" ]]; then
    python -c "import sys; assert hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled(), 'expected a free-threaded interpreter with the GIL disabled'"
elif [[ "$PYTHON_RC" == "true" ]]; then
    python -c "import sys, sysconfig; assert sysconfig.get_config_var('Py_GIL_DISABLED') in (0, None) and sys._is_gil_enabled(), 'expected a GIL-enabled Python RC'"
fi

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper.openmp_helpers_inner

if [[ "$PYTHON_FREETHREADED" == "true" ]]; then
    python -c "import sys, numpy, scipy.linalg, tests._openmp_test_helper.openmp_helpers_inner as _; assert not sys._is_gil_enabled(), 'GIL was re-enabled after importing native extensions'"
fi

# Introspect API scope:
PYTHONPATH=. python -X faulthandler tests/empirical_scope_observation.py blas
PYTHONPATH=. python -X faulthandler tests/empirical_scope_observation.py openmp

pytest -vlrxXs -W error -k "$TESTS" --junitxml=test_result.xml --cov=threadpoolctl --cov-report xml
