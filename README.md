# Thread-pool Controls [![Build Status](https://dev.azure.com/karementmoi/loky/_apis/build/status/joblib.threadpoolctl?branchName=master)](https://dev.azure.com/karementmoi/loky/_build/latest?definitionId=2&branchName=master) [![codecov](https://codecov.io/gh/joblib/threadpoolctl/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/threadpoolctl)

Python helpers to limit the number of threads used in the
threadpool-backed of common native libraries used for scientific
computing and data science (e.g. BLAS and OpenMP).

Fine control of the underlying thread-pool size can be useful in
workloads that involve nested parallelism so as to mitigate
oversubscription issues.

## Installation

- For users, you can install the last published version from PyPI:

```
pip install threadpoolctl
```

- For contributors, you can install from the source repository in
  developer mode:

```
pip install -r dev-requirements.txt
flit install --symlink
```

then you can run the tests with pytest:

```
pytest
```

## Usage

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

## Maintainers

To make a release:

```
pip install flit
flit build
```

Check the contents of `dist/`.

If everything is fine, make a commit for the release, tag it, push the
tag to github and then:

```
flit publish
```
