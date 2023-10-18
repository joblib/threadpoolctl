@rem https://github.com/numba/numba/blob/master/buildscripts/incremental/setup_conda_environment.cmd
@rem The cmd /C hack circumvents a regression where conda installs a conda.bat
@rem script in non-root environments.
set CONDA_INSTALL=cmd /C conda install -q -y
set PIP_INSTALL=pip install -q

@echo on

@rem Deactivate any environment
call deactivate
@rem Clean up any left-over from a previous build and install version of python
conda update -n base conda conda-libmamba-solver -q -y
conda config --set solver libmamba
conda remove --all -q -y -n %VIRTUALENV%
conda create -n %VIRTUALENV% -q -y python=%PYTHON_VERSION%

call activate %VIRTUALENV%
python -m pip install -U pip
python --version
pip --version

@rem Install dependencies with either conda or pip.
set TO_INSTALL=numpy scipy cython pytest

if "%PACKAGER%" == "conda" (%CONDA_INSTALL% %TO_INSTALL%)
if "%PACKAGER%" == "conda-forge" (%CONDA_INSTALL% -c conda-forge %TO_INSTALL% blas[build=%BLAS%])
if "%PACKAGER%" == "pip" (%PIP_INSTALL% %TO_INSTALL%)

@rem Install extra developer dependencies
pip install -q -r dev-requirements.txt

@rem Install package
flit install --symlink

@rem Build the cython test helper for openmp
bash ./continuous_integration/build_test_ext.sh

if %errorlevel% neq 0 exit /b %errorlevel%
