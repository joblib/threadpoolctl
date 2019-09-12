1.1.0 (2019-09-12)
==================

- Detect libraries referenced by symlinks (e.g. BLAS libraries from
  conda-forge).
  https://github.com/joblib/threadpoolctl/pull/34

- New method `get_original_num_threads` on the `threadpool_limits`
  context manager to cheaply access the initial state of the runtime.
  https://github.com/joblib/threadpoolctl/pull/32

- Add support for BLIS.
  https://github.com/joblib/threadpoolctl/pull/23


1.0.0 (2019-06-03)
==================

Initial release.
