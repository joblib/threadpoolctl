call activate %VIRTUALENV%

# Display version information
python -m pip list

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper.openmp_helpers_inner

pytest -vlrxXs --junitxml=%JUNITXML% --cov=threadpoolctl
