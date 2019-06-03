call activate %VIRTUALENV%

pytest -vlrxXs --junitxml=%JUNITXML% --cov=threadpoolctl
