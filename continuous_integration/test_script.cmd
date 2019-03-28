call activate %VIRTUALENV%

pytest -vlx --junitxml=%JUNITXML% \
    --cov=threadpoolctl --cov-config=%COVERAGE_DATA%
