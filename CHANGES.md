2.1.0 (2020-05-29)
==================

- New commandline interface:

      python -m threadpoolctl -i numpy

  will try to import the `numpy` package and then return the output of
  `threadpoolctl.threadpool_info()` on STDOUT formatted using the JSON
  syntax. This makes it easier to quickly introspect a Python environment.


2.0.0 (2019-12-05)
==================

- Expose MKL, BLIS and OpenBLAS threading layer in informations displayed by
  `threadpool_info`. This information is referenced in the `threading_layer`
  field.
  https://github.com/joblib/threadpoolctl/pull/48
  https://github.com/joblib/threadpoolctl/pull/60

- When threadpoolctl finds libomp (LLVM OpenMP) and libiomp (Intel OpenMP)
  both loaded, a warning is raised to recall that using threadpoolctl with
  this mix of OpenMP libraries may cause crashes or deadlocks.
  https://github.com/joblib/threadpoolctl/pull/49

1.1.0 (2019-09-12)
==================

- Detect libraries referenced by symlinks (e.g. BLAS libraries from
  conda-forge).
  https://github.com/joblib/threadpoolctl/pull/34

- Add support for BLIS.
  https://github.com/joblib/threadpoolctl/pull/23

- Breaking change: method `get_original_num_threads` on the `threadpool_limits`
  context manager to cheaply access the initial state of the runtime:
    - drop the `user_api` parameter;
    - instead return a dict `{user_api: num_threads}`;
    - fixed a bug when the limit parameter of `threadpool_limits` was set to
      `None`.

  https://github.com/joblib/threadpoolctl/pull/32


1.0.0 (2019-06-03)
==================

Initial release.
