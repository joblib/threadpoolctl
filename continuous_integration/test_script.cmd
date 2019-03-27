set DEFAULT_PYTEST_ARGS=-vlx

call activate %VIRTUALENV%

pytest --junitxml=%JUNITXML% %DEFAULT_PYTEST_ARGS%
