call activate %VIRTUALENV%

pytest -vl --junitxml=%JUNITXML% --cov=threadpoolctl
