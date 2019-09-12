# Thread-pool Controls [![Build Status](https://dev.azure.com/joblib/threadpoolctl/_apis/build/status/joblib.threadpoolctl?branchName=master)](https://dev.azure.com/joblib/threadpoolctl/_build/latest?definitionId=1&branchName=master) [![codecov](https://codecov.io/gh/joblib/threadpoolctl/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/threadpoolctl)

Python helpers to limit the number of threads used in the
threadpool-backed of common native libraries used for scientific
computing and data science (e.g. BLAS and OpenMP).

Fine control of the underlying thread-pool size can be useful in
workloads that involve nested parallelism so as to mitigate
oversubscription issues.

## Installation

- For users, install the last published version from PyPI:

  ```bash
  pip install threadpoolctl
  ```

- For contributors, install from the source repository in developer
  mode:

  ```bash
  pip install -r dev-requirements.txt
  flit install --symlink
  ```

  then you run the tests with pytest:

  ```bash
  pytest
  ```

## Usage

### Runtime Introspection

Introspect the current state of the threadpool-enabled runtime libraries
that are loaded when importing Python packages:

```python
>>> from threadpoolctl import threadpool_info
>>> from pprint import pprint
>>> pprint(threadpool_info())
[]

>>> import numpy
>>> pprint(threadpool_info())
[{'filepath': '/opt/venvs/py37/lib/python3.7/site-packages/numpy/.libs/libopenblasp-r0-382c8f3a.3.5.dev.so',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'user_api': 'blas',
  'version': '0.3.5.dev'}]

>>> import xgboost
>>> pprint(threadpool_info())
[{'filepath': '/opt/venvs/py37/lib/python3.7/site-packages/numpy/.libs/libopenblasp-r0-382c8f3a.3.5.dev.so',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'user_api': 'blas',
  'version': '0.3.5.dev'},
 {'filepath': '/opt/venvs/py37/lib/python3.7/site-packages/scipy/.libs/libopenblasp-r0-8dca6697.3.0.dev.so',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'user_api': 'blas',
  'version': None},
 {'filepath': '/usr/lib/x86_64-linux-gnu/libgomp.so.1',
  'internal_api': 'openmp',
  'num_threads': 4,
  'prefix': 'libgomp',
  'user_api': 'openmp',
  'version': None}]
```

### Set the maximum size of thread-pools

Control the number of threads used by the underlying runtime libraries
in specific sections of your Python program:

```python
from threadpoolctl import threadpool_limits
import numpy as np


with threadpool_limits(limits=1, user_api='blas'):
    # In this block, calls to blas implementation (like openblas or MKL)
    # will be limited to use only one thread. They can thus be used jointly
    # with thread-parallelism.
    a = np.random.randn(1000, 1000)
    a_squared = a @ a
```

### Known limitation

`threadpool_limits` does not act as expected in nested parallel loops
managed by distinct OpenMP runtime implementations (for instance libgomp
from GCC and libomp from clang/llvm or libiomp from ICC).

See the `test_openmp_nesting()` function in `tests/test_threadpoolctl.py`
for an example.

## Maintainers

To make a release:

Bump the version number (`__version__`) in `threadpoolctl.py`.

Build the distribution archives:

```bash
pip install flit
flit build
```

Check the contents of `dist/`.

If everything is fine, make a commit for the release, tag it, push the
tag to github and then:

```bash
flit publish
```
