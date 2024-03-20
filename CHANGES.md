3.4.0 (2024-03-20)
==================

- Added support for Python interpreters statically linked against libc or linked against
  alternative implementations of libc like musl (on Alpine Linux for instance).
  https://github.com/joblib/threadpoolctl/pull/171

- Added support for Pyodide
  https://github.com/joblib/threadpoolctl/pull/169

3.3.0 (2024-02-14)
==================

- Extended FlexiBLAS support to be able to switch backend at runtime.
  https://github.com/joblib/threadpoolctl/pull/163

- Added support for FlexiBLAS
  https://github.com/joblib/threadpoolctl/pull/156

- Fixed a bug where an unsupported library would be detected because it shares a common
  prefix with one of the supported libraries. Now the symbols are also checked to
  identify the supported libraries.
  https://github.com/joblib/threadpoolctl/pull/151

3.2.0 (2023-07-13)
==================

- Dropped support for Python 3.6 and 3.7.

- Added support for custom library controllers. Custom controllers must inherit from
  the `threadpoolctl.LibController` class and be registered to threadpoolctl using the
  `threadpoolctl.register` function.
  https://github.com/joblib/threadpoolctl/pull/138

- A warning is raised on macOS when threadpoolctl finds both Intel OpenMP and LLVM
  OpenMP runtimes loaded simultaneously by the same Python program. See details and
  workarounds at https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md.
  https://github.com/joblib/threadpoolctl/pull/142

3.1.0 (2022-01-31)
==================

- Fixed a detection issue of the BLAS libraires packaged by conda-forge on Windows.
  https://github.com/joblib/threadpoolctl/pull/112

- `threadpool_limits` and `ThreadpoolController.limit` now accept the string
  "sequential_blas_under_openmp" for the `limits` parameter. It should only be used for
  the specific case when one wants to have sequential BLAS calls within an OpenMP
  parallel region. It takes into account the unexpected behavior of OpenBLAS with the
  OpenMP threading layer.
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
