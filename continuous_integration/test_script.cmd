set DEFAULT_PYTEST_ARGS=-vlx --cov=threadpoolctl

call activate %VIRTUALENV%

pytest --junitxml=%JUNITXML% %DEFAULT_PYTEST_ARGS%
