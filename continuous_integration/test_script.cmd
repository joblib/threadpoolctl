call activate %VIRTUALENV%

pytest -vlx --junitxml=%JUNITXML% --cov=threadpoolctl
