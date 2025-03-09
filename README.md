# Thread-pool Controls [![Build Status](https://github.com/joblib/threadpoolctl/actions/workflows/test.yml/badge.svg?branch=master)](https://dev.azure.com/joblib/threadpoolctl/_build/latest?definitionId=1&branchName=master) [![codecov](https://codecov.io/gh/joblib/threadpoolctl/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/threadpoolctl)

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

### Command Line Interface

Get a JSON description of thread-pools initialized when importing python
packages such as numpy or scipy for instance:

```
python -m threadpoolctl -i numpy scipy.linalg
[
  {
    "filepath": "/home/ogrisel/miniconda3/envs/tmp/lib/libmkl_rt.so",
    "prefix": "libmkl_rt",
    "user_api": "blas",
    "internal_api": "mkl",
    "version": "2019.0.4",
    "num_threads": 2,
    "threading_layer": "intel"
  },
  {
    "filepath": "/home/ogrisel/miniconda3/envs/tmp/lib/libiomp5.so",
    "prefix": "libiomp",
    "user_api": "openmp",
    "internal_api": "openmp",
    "version": null,
    "num_threads": 4
  }
]
```

The JSON information is written on STDOUT. If some of the packages are missing,
a warning message is displayed on STDERR.

### Python Runtime Programmatic Introspection

Introspect the current state of the threadpool-enabled runtime libraries
that are loaded when importing Python packages:

```python
>>> from threadpoolctl import threadpool_info
>>> from pprint import pprint
>>> pprint(threadpool_info())
[]

>>> import numpy
>>> pprint(threadpool_info())
[{'filepath': '/home/ogrisel/miniconda3/envs/tmp/lib/libmkl_rt.so',
  'internal_api': 'mkl',
  'num_threads': 2,
  'prefix': 'libmkl_rt',
  'threading_layer': 'intel',
  'user_api': 'blas',
  'version': '2019.0.4'},
 {'filepath': '/home/ogrisel/miniconda3/envs/tmp/lib/libiomp5.so',
  'internal_api': 'openmp',
  'num_threads': 4,
  'prefix': 'libiomp',
  'user_api': 'openmp',
  'version': None}]

>>> import xgboost
>>> pprint(threadpool_info())
[{'filepath': '/home/ogrisel/miniconda3/envs/tmp/lib/libmkl_rt.so',
  'internal_api': 'mkl',
  'num_threads': 2,
  'prefix': 'libmkl_rt',
  'threading_layer': 'intel',
  'user_api': 'blas',
  'version': '2019.0.4'},
 {'filepath': '/home/ogrisel/miniconda3/envs/tmp/lib/libiomp5.so',
  'internal_api': 'openmp',
  'num_threads': 4,
  'prefix': 'libiomp',
  'user_api': 'openmp',
  'version': None},
 {'filepath': '/home/ogrisel/miniconda3/envs/tmp/lib/libgomp.so.1.0.0',
  'internal_api': 'openmp',
  'num_threads': 4,
  'prefix': 'libgomp',
  'user_api': 'openmp',
  'version': None}]
```

In the above example, `numpy` was installed from the default anaconda channel and comes
with MKL and its Intel OpenMP (`libiomp5`) implementation while `xgboost` was installed
from pypi.org and links against GNU OpenMP (`libgomp`) so both OpenMP runtimes are
loaded in the same Python program.

The state of these libraries is also accessible through the object oriented API:

```python
>>> from threadpoolctl import ThreadpoolController, threadpool_info
>>> from pprint import pprint
>>> import numpy
>>> controller = ThreadpoolController()
>>> pprint(controller.info())
[{'architecture': 'Haswell',
  'filepath': '/home/jeremie/miniconda/envs/dev/lib/libopenblasp-r0.3.17.so',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'threading_layer': 'pthreads',
  'user_api': 'blas',
  'version': '0.3.17'}]

>>> controller.info() == threadpool_info()
True
```

### Setting the Maximum Size of Thread-Pools

Control the number of threads used by the underlying runtime libraries
in specific sections of your Python program:

```python
>>> from threadpoolctl import threadpool_limits
>>> import numpy as np

>>> with threadpool_limits(limits=1, user_api='blas'):
...     # In this block, calls to blas implementation (like openblas or MKL)
...     # will be limited to use only one thread. They can thus be used jointly
...     # with thread-parallelism.
...     a = np.random.randn(1000, 1000)
...     a_squared = a @ a
```

The threadpools can also be controlled via the object oriented API, which is especially
useful to avoid searching through all the loaded shared libraries each time. It will
however not act on libraries loaded after the instantiation of the
`ThreadpoolController`:

```python
>>> from threadpoolctl import ThreadpoolController
>>> import numpy as np
>>> controller = ThreadpoolController()

>>> with controller.limit(limits=1, user_api='blas'):
...     a = np.random.randn(1000, 1000)
...     a_squared = a @ a
```

### Restricting the limits to the scope of a function

`threadpool_limits` and `ThreadpoolController` can also be used as decorators to set
the maximum number of threads used by the supported libraries at a function level. The
decorators are accessible through their `wrap` method:

```python
>>> from threadpoolctl import ThreadpoolController, threadpool_limits
>>> import numpy as np
>>> controller = ThreadpoolController()

>>> @controller.wrap(limits=1, user_api='blas')
... # or @threadpool_limits.wrap(limits=1, user_api='blas')
... def my_func():
...     # Inside this function, calls to blas implementation (like openblas or MKL)
...     # will be limited to use only one thread.
...     a = np.random.randn(1000, 1000)
...     a_squared = a @ a
...
```

### Switching the FlexiBLAS backend

`FlexiBLAS` is a BLAS wrapper for which the BLAS backend can be switched at runtime.
`threadpoolctl` exposes python bindings for this feature. Here's an example but note
that this part of the API is experimental and subject to change without deprecation:

```python
>>> from threadpoolctl import ThreadpoolController
>>> import numpy as np
>>> controller = ThreadpoolController()

>>> controller.info()
[{'user_api': 'blas',
  'internal_api': 'flexiblas',
  'num_threads': 1,
  'prefix': 'libflexiblas',
  'filepath': '/usr/local/lib/libflexiblas.so.3.3',
  'version': '3.3.1',
  'available_backends': ['NETLIB', 'OPENBLASPTHREAD', 'ATLAS'],
  'loaded_backends': ['NETLIB'],
  'current_backend': 'NETLIB'}]

# Retrieve the flexiblas controller
>>> flexiblas_ct = controller.select(internal_api="flexiblas").lib_controllers[0]

# Switch the backend with one predefined at build time (listed in "available_backends")
>>> flexiblas_ct.switch_backend("OPENBLASPTHREAD")
>>> controller.info()
[{'user_api': 'blas',
  'internal_api': 'flexiblas',
  'num_threads': 4,
  'prefix': 'libflexiblas',
  'filepath': '/usr/local/lib/libflexiblas.so.3.3',
  'version': '3.3.1',
  'available_backends': ['NETLIB', 'OPENBLASPTHREAD', 'ATLAS'],
  'loaded_backends': ['NETLIB', 'OPENBLASPTHREAD'],
  'current_backend': 'OPENBLASPTHREAD'},
 {'user_api': 'blas',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'filepath': '/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.8.so',
  'version': '0.3.8',
  'threading_layer': 'pthreads',
  'architecture': 'Haswell'}]

# It's also possible to directly give the path to a shared library
>>> flexiblas_controller.switch_backend("/home/jeremie/miniforge/envs/flexiblas_threadpoolctl/lib/libmkl_rt.so")
>>> controller.info()
[{'user_api': 'blas',
  'internal_api': 'flexiblas',
  'num_threads': 2,
  'prefix': 'libflexiblas',
  'filepath': '/usr/local/lib/libflexiblas.so.3.3',
  'version': '3.3.1',
  'available_backends': ['NETLIB', 'OPENBLASPTHREAD', 'ATLAS'],
  'loaded_backends': ['NETLIB',
   'OPENBLASPTHREAD',
   '/home/jeremie/miniforge/envs/flexiblas_threadpoolctl/lib/libmkl_rt.so'],
  'current_backend': '/home/jeremie/miniforge/envs/flexiblas_threadpoolctl/lib/libmkl_rt.so'},
 {'user_api': 'openmp',
  'internal_api': 'openmp',
  'num_threads': 4,
  'prefix': 'libomp',
  'filepath': '/home/jeremie/miniforge/envs/flexiblas_threadpoolctl/lib/libomp.so',
  'version': None},
 {'user_api': 'blas',
  'internal_api': 'openblas',
  'num_threads': 4,
  'prefix': 'libopenblas',
  'filepath': '/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.8.so',
  'version': '0.3.8',
  'threading_layer': 'pthreads',
  'architecture': 'Haswell'},
 {'user_api': 'blas',
  'internal_api': 'mkl',
  'num_threads': 2,
  'prefix': 'libmkl_rt',
  'filepath': '/home/jeremie/miniforge/envs/flexiblas_threadpoolctl/lib/libmkl_rt.so.2',
  'version': '2024.0-Product',
  'threading_layer': 'gnu'}]
```

You can observe that the previously linked OpenBLAS shared object stays loaded by
the Python program indefinitely, but FlexiBLAS itself no longer delegates BLAS calls
to OpenBLAS as indicated by the `current_backend` attribute.
### Writing a custom library controller

Currently, `threadpoolctl` has support for `OpenMP` and the main `BLAS` libraries.
However it can also be used to control the threadpool of other native libraries,
provided that they expose an API to get and set the limit on the number of threads.
For that, one must implement a controller for this library and register it to
`threadpoolctl`.

A custom controller must be a subclass of the `LibController` class and implement
the attributes and methods described in the docstring of `LibController`. Then this
new controller class must be registered using the `threadpoolctl.register` function.
An complete example can be found [here](
  https://github.com/joblib/threadpoolctl/blob/master/tests/_pyMylib/__init__.py).

### Sequential BLAS within OpenMP parallel region

When one wants to have sequential BLAS calls within an OpenMP parallel region, it's
safer to set `limits="sequential_blas_under_openmp"` since setting `limits=1` and
`user_api="blas"` might not lead to the expected behavior in some configurations
(e.g. OpenBLAS with the OpenMP threading layer
https://github.com/xianyi/OpenBLAS/issues/2985).

### Known Limitations

- `threadpool_limits` can fail to limit the number of inner threads when nesting
  parallel loops managed by distinct OpenMP runtime implementations (for instance
  libgomp from GCC and libomp from clang/llvm or libiomp from ICC).

  See the `test_openmp_nesting` function in [tests/test_threadpoolctl.py](
  https://github.com/joblib/threadpoolctl/blob/master/tests/test_threadpoolctl.py)
  for an example. More information can be found at:
  https://github.com/jeremiedbb/Nested_OpenMP

  Note however that this problem does not happen when `threadpool_limits` is
  used to limit the number of threads used internally by BLAS calls that are
  themselves nested under OpenMP parallel loops. `threadpool_limits` works as
  expected, even if the inner BLAS implementation relies on a distinct OpenMP
  implementation.

- Using Intel OpenMP (ICC) and LLVM OpenMP (clang) in the same Python program
  under Linux is known to cause problems. See the following guide for more details
  and workarounds:
  https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md

- Setting the maximum number of threads of the OpenMP and BLAS libraries has a global
  effect and impacts the whole Python process. There is no thread level isolation as
  these libraries do not offer thread-local APIs to configure the number of threads to
  use in nested parallel calls.


## Maintainers

To make a release:

- Bump the version number (`__version__`) in `threadpoolctl.py` and update the
  release date in `CHANGES.md`.

- Build the distribution archives:

```bash
pip install flit
flit build
```

and check the contents of `dist/`.

- If everything is fine, make a commit for the release, tag it and push the
tag to github:

```bash
git tag -a X.Y.Z
git push git@github.com:joblib/threadpoolctl.git X.Y.Z
```

- Upload the wheels and source distribution to PyPI using flit. Since PyPI doesn't
  allow password authentication anymore, the username needs to be changed to the
  generic name `__token__`:

```bash
FLIT_USERNAME=__token__ flit publish
```

  and a PyPI token has to be passed in place of the password.

- Create a PR for the release on the [conda-forge feedstock](https://github.com/conda-forge/threadpoolctl-feedstock) (or wait for the bot to make it).

- Publish the release on github.

### Credits

The initial dynamic library introspection code was written by @anton-malakhov
for the smp package available at https://github.com/IntelPython/smp .

threadpoolctl extends this for other operating systems. Contrary to smp,
threadpoolctl does not attempt to limit the size of Python multiprocessing
pools (threads or processes) or set operating system-level CPU affinity
constraints: threadpoolctl only interacts with native libraries via their
public runtime APIs.
