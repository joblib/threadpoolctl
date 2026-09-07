# Thread-pool Controls [![Build Status](https://github.com/joblib/threadpoolctl/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/joblib/threadpoolctl/actions?query=branch%3Amaster) [![codecov](https://codecov.io/gh/joblib/threadpoolctl/branch/master/graph/badge.svg)](https://codecov.io/gh/joblib/threadpoolctl)

Python helpers to limit the number of threads used in the
threadpool-backed of common native libraries used for scientific
computing and data science (e.g. BLAS and OpenMP).

Fine control of the underlying thread-pool size can be useful in
workloads that involve nested parallelism so as to mitigate
oversubscription issues.

Note that "limiting the number of threads" in practice involves a variety of
potential semantics depending on the underlying third-party library; see the
section on [semantics](#semantics) below.

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

## Usage: Introspection and debugging

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

## Usage when not using Python threads: Restricting Controlled Library Thread Pool Sizes

There are two scenarios in which you might want to use `threadpoolctl`; each
requires you to use different APIs.

1. You do not expect to use any Python threads, so all the work will be started
   directly from the main thread in the process. This is a simple case
   where we can globally set thread limits.
2. You will be parallelizing work using a Python thread pool, and your goal is
   therefore to limit controlled libraries' thread pool sizes when
   concurrently called from Python threads. This case is a bit more
   complex to handle properly and requires a bit more verbose code.

This section will cover the former case, and the latter is covered in the next
usage section.

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

### Restricting the Limits to the Scope of a Function

`threadpool_limits` and `ThreadpoolController` can also be used as decorators to set
the maximum number of threads used by the supported libraries at a function level. The
decorators are accessible through their `wrap` method.

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

## Usage for Python threads: Restricting Controlled Library Thread Pool Sizes

This section covers APIs to use when you will be using Python thread pools to parallelize work.

### Setting the Maximum Size of Thread-Pools, When Python Thread Pools Are Used

Limiting thread pool size in controlled libraries requires a two-step process.
**Importantly, each Python worker thread must also call a method to limit
controlled libraries in that thread.** With Python's
`concurrent.futures.ThreadPoolExecutor`, you can do so by passing in an
initializer function that will get called on thread startup.

```python
from threadpoolctl import threadpool_limits
from concurrent.futures import ThreadPoolExecutor

# This top-level limiter doesn't actually change the limits initially; it is
# there to ensure the limits are reset _after_ the Python thread pool is done.
# This is necessary because some underlying limiting APIs operate on a
# process-wide basis.
with threadpool_limits():
    # Make sure each Python worker thread also calls threadpool_limits(). If
    # you're using another thread pool class, you will need to do so some other
    # way.
    with ThreadPoolExecutor(4, initializer=lambda: threadpool_limits(limits=1)) as pool:
        # ... run some BLAS-using code in the thread pool ...
        pool.map(somefunc, someargs)
```

To prevent loading shared libraries repeatedly, you can reuse a
`ThreadpoolController` object:

```python
from threadpoolctl import ThreadpoolController

# This won't have any side-effects:
CONTROLLER = ThreadpoolController()

with (
    CONTROLLER.limit(),
    ThreadPoolExecutor(4, initializer=lambda: CONTROLLER.limit(limits=1)) as pool,
):
    # ... run some BLAS-using code in the thread pool ...
    pool.map(somefunc, someargs)

# Later...
with (
    CONTROLLER.limit(),
    ThreadPoolExecutor(4, initializer=lambda: CONTROLLER.limit(limits=2)) as pool,
):
    # ... run some BLAS-using code in the thread pool ...
    pool.map(somefunc, someargs)
```

You can also operate without a context manager:

```python
from threadpoolctl import ThreadpoolController

CONTROLLER = ThreadpoolController()
try:
    limiter = CONTROLLER.limit()
    with ThreadPoolExecutor(
            4, initializer=lambda: CONTROLLER.limit(limits=1)) as pool:
        # ... run some BLAS-using code in the thread pool ...
        pool.map(somefunc, someargs)
finally:
    limiter.restore_original_limits()

```

### Switching Back And Forth Between Main Thread and Python Threads

Unfortunately not all controlled libraries providing limiting APIs that are
thread-specific, as detailed in the section on [semantics](#semantics) below.
Limiting some libraries' thread pool sizes can therefore impact the whole
process. This makes switching back and forth between running code that
uses these libraries in Python threads and running it in the main thread a bit
more complex: you need to set the limits each time you switch back and forth.

Let's say your computer has 4 cores, and you're using some OpenMP API.

```python
POOL = ThreadPoolExecutor(4)
CONTROLLER = ThreadpoolController()

# 1. Run some work in a Python thread pool (OpenMP effectively disabled).
with CONTROLLER.limit(limits=1) as limiter:

    def limit_then_do_work(*args, **kwargs):
        # Set a limit on OpenMP in the current thread:
        CONTROLLER.limit(limits=1)
        # Do the actual work:
        return do_real_work_with_openmp(*args, **kwargs)

    results = POOL.map(limit_then_do_work, args)


# 2. Run some work with 4-threads OpenMP parallelism:
with CONTROLLER.limit(limits=4):
    results2 = do_more_work_with_openmp(results)


# 3. Nest some OpenMP parallelism under Python-level parallelism:
with CONTROLLER.limit(limits=2) as limiter:

    def limit_then_do_work2(*args, **kwargs):
        limiter.limit(limits=2)
        return do_even_more_real_work_with_openmp(*args, **kwargs)

    results3 = POOL.map(limit_then_do_work2, results2)
```

## Usage: Additional APIs and details

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

- Setting the maximum number of threads of the OpenMP and BLAS libraries has
  inconsistent scope and semantics (thread-local vs process-wide) depending on
  the underlying library. For more details see
  https://github.com/joblib/threadpoolctl/issues/208

  For example, if you're using OpenMP with libgomp (gcc) or libomp (clang**, the
  setting is thread-local and sets how many OpenMP threads will be started in
  the current thread. On the other hand, with OpenBLAS with pthreads backend or
  on Windows, the setting is process-wide and impacts the size of a process-wide
  thread pool shared across all threads in the process.

## <div id="semantics"> Semantics of thread limiting </div>

Setting the number of threads may seem like a simple operation, but in practice
it can do quite different things depending on the underlying third-party library
being limited.

### Kind of thread pool

* In some libraries, there is a shared process-wide thread pool. Setting the
  number of threads changes the size of this shared pool.
* In other libraries, there is a thread pool per calling thread. So each Python
  thread gets its own personal thread pool from the third-party library.

To give a concrete example: if you have 10 cores, and you're using OpenBLAS with
the `pthreads` backend on Linux, by default there is a single 10-worker pool of
threads created by OpenBLAS. If you start 10 Python threads, each calling BLAS
routines, all those Python threads will feed into that single 10-thread pool.
In total, you will have 20 threads running (10 Python, 10 OpenBLAS).

On the other hand, if you use MKL, each Python thread gets its own individual
pool of worker threads from MKL, so if each Python threads calls a BLAS routine
in MKL, you will get 10×10 = 100 MKL threads!

### Scope of limiting

If you have a third-party library that creates a thread pool per calling thread,
there is another question about what limiting the number of threads means: which
threads are affected by the limiting API?

* **All threads in the process:** In this case, setting the limits changes it
  for all threads in the process. So if you limit from the main Python thread,
  all other Python threads will have the same limit.
* **Current thread only:** Only the current thread's thread pool size will be
  limited.

MKL for example has two APIs, covering both cases,
[`mkl_set_num_threads()`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-set-num-threads.html)
and [`mkl_set_num_threads_local()`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/mkl-set-num-threads-local.html) respectively.

### `threadpoolctl`'s policy for thread limiting

When there is a choice between different APIs, `threadpoolctl` will prefer APIs
that:

1. Have a per-thread worker pool (so each Python thread gets its own pool from
   the underlying third-party library)
2. Only affect the current thread.

Current libraries where `threadpoolctl` explicitly makes this choice are:

* MKL, for all threading backends.
* OpenBLAS (v0.3.34 or later) when using the OpenMP backend, on Linux and macOS.

When using OpenMP, this is also the default on Linux and macOS. On Windows
OpenMP has a per-thread worker pool but the limiting API affects all threads in
the process, not just the current one.

### Checking the behavior of your installed libraries

When you run `python -m threadpoolctl` it will include the scope of the API
limit in the `"thread_limit_scope"` field, with the value of `"process"`
indicating the API affects the whole process and `"current_thread"` indicating a
per-thread thread pool. Information about the kind of limiting in the former
case (shared process-wide pool, or per-thread pool) is not included.

Here is example output for OpenBLAS with pthreads:

```shell-session
$ python -m threadpoolctl -i numpy
[
  {
    "user_api": "blas",
    "internal_api": "openblas",
    "num_threads": 12,
    "prefix": "libscipy_openblas",
    "version": "0.3.33.112.0",
    "threading_layer": "pthreads",
    "architecture": "Haswell",
    "thread_limit_scope": "process"
  }
]
```

You can also use another script to empirically determine both the kind and scope of the API. From a checkout of the [`threadpoolctl` GitHub repo](https://github.com/joblib/threadpoolctl):

```shell-session
$ python -m tests.empirical_scope_observation blas
== Observed behavior: blas ==
10x12 Python threads each running a parallel workload with a 2 thread limit.
Maximum number of observed threads (includes Python and blas threads): 53
Presumed API scope: process-wide shared thread pool
```

You can also call this script with an `openmp` argument instead of `blas`.

## Maintainers

To make a release:

- Create a PR to bump the version number (`__version__`) in `threadpoolctl.py` and
  update the release date in `CHANGES.md`.

- Merge the PR and check that the `Publish threadpoolctl distribution to TestPyPI` job
  of the `publish-to-pypi.yml` workflow successfully uploaded the wheel and source
  distribution to Test PyPI.

- If everything is fine create a tag for the release and push it to github:

  ```bash
  git tag -a X.Y.Z
  git push git@github.com:joblib/threadpoolctl.git X.Y.Z
  ```

- Check that the `Publish threadpoolctl distribution to PyPI` job of the
  `publish-to-pypi.yml` workflow successfully uploaded the wheel and source distribution
  to PyPI this time.

- Create a PR for the release on the [conda-forge feedstock](https://github.com/conda-forge/threadpoolctl-feedstock) (or wait for the bot to make it).

- Publish the release on github.

If for some reason the steps above can't be achieved and a munual upload of the wheel
and source distribution is needed:

- Build the distribution archives:

  ```bash
  pip install flit
  flit build
  ```

- Upload the wheels and source distribution to PyPI using flit. Since PyPI doesn't
  allow password authentication anymore, the username needs to be changed to the
  generic name `__token__`:

  ```bash
  FLIT_USERNAME=__token__ flit publish
  ```

  and a PyPI token has to be passed in place of the password.

### Credits

The initial dynamic library introspection code was written by @anton-malakhov
for the smp package available at https://github.com/IntelPython/smp .

threadpoolctl extends this for other operating systems. Contrary to smp,
threadpoolctl does not attempt to limit the size of Python multiprocessing
pools (threads or processes) or set operating system-level CPU affinity
constraints: threadpoolctl only interacts with native libraries via their
public runtime APIs.
