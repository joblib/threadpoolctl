call activate %VIRTUALENV%

# Use the CLI to display the effective runtime environment prior to
# launching the tests:
python -m threadpoolctl -i numpy scipy.linalg tests._openmp_test_helper

pytest -vlrxXs --junitxml=%JUNITXML% --cov=threadpoolctl
