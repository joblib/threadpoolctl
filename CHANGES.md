3.1.0 (TBD)
===========

- Fixed a detection issue of the BLAS libraires packaged by conda-forge on Windows.
  https://github.com/joblib/threadpoolctl/pull/112

- New helper function `threadpoolctl.get_params_for_sequential_blas_under_openmp` and
  new method `ThreadpoolController.get_params_for_sequential_blas_under_openmp` that
  returns the appropriate params to pass to `threadpool_info` or
  `ThreadpoolController.limit` for the specific case when one wants to have sequential
  BLAS calls within an OpenMP parallel region. This helper function takes into account
  the unexpected behavior of OpenBLAS with the OpenMP threading layer.
  https://github.com/joblib/threadpoolctl/pull/114

3.0.0 (2021-10-01)
==================

- New object `threadpooctl.ThreadpoolController` which holds controllers for all the
  supported native libraries. The states of these libraries is accessible through the
  `info` method (equivalent to `threadpoolctl.threadpool_info()`) and their number of
  threads can be limited with the `limit` method which can be used as a context
  manager (equivalent to `threadpoolctl.threadpool_limits()`). This is especially useful
  to avoid searching through all loaded shared libraries each time.
  https://github.com/joblib/threadpoolctl/pull/95

- Added support for OpenBLAS built for 64bit integers in Fortran.
  https://github.com/joblib/threadpoolctl/pull/101

- Added the possibility to use `threadpoolctl.threadpool_limits` and
  `threadpooctl.ThreadpoolController` as decorators through their `wrap` method.
  https://github.com/joblib/threadpoolctl/pull/102

- Fixed an attribute error when using old versions of OpenBLAS or BLIS that are
  missing version query functions.
  https://github.com/joblib/threadpoolctl/pull/88
  https://github.com/joblib/threadpoolctl/pull/91

- Fixed an attribute error when python is run with -OO.
  https://github.com/joblib/threadpoolctl/pull/87

2.2.0 (2021-07-09)
==================

- `threadpoolctl.threadpool_info()` now reports the architecture of the CPU
  cores detected by OpenBLAS (via `openblas_get_corename`) and BLIS (via
  `bli_arch_query_id` and `bli_arch_string`).

- Fixed a bug when the version of MKL was not found. The
  "version" field is now set to None in that case.
  https://github.com/joblib/threadpoolctl/pull/82

2.1.0 (2020-05-29)
==================

- New commandline interface:

      python -m threadpoolctl -i numpy

  will try to import the `numpy` package and then return the output of
  `threadpoolctl.threadpool_info()` on STDOUT formatted using the JSON
  syntax. This makes it easier to quickly introspect a Python environment.


2.0.0 (2019-12-05)
==================

- Expose MKL, BLIS and OpenBLAS threading layer in information displayed by
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
