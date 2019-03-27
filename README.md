# Thread-pool Controls [![Build Status](https://dev.azure.com/karementmoi/loky/_apis/build/status/joblib.threadpoolctl?branchName=master)](https://dev.azure.com/karementmoi/loky/_build/latest?definitionId=2&branchName=master) [![codecov](https://codecov.io/gh/joblib/threadpoolctl/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/threadpoolctl)



Python helpers to limit the number of threads used in threadpool-backed parallelism for C-libraries.


# Installation

```
pip install threadpoolctl
```

# Usage

```python
from threadpoolctl import threadpool_limits


with threadpool_limits(limits=1, user_api='blas'):
    # In this block, calls to blas implementation (like openblas or MKL)
    # will be limited to use only one thread. They can thus be used jointly
    # with thread-parallelism or openmp calls.
    ...
```
