call activate %VIRTUALENV%

python continuous_integration/display_versions.py

pytest -vlrxXs --junitxml=%JUNITXML% --cov=threadpoolctl
