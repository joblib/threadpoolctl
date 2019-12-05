2.0.0 (TBD)
===========

- Expose MKL, BLIS and OpenBLAS threading layer in informations displayed by
  `threadpool_info`. This information is referenced in the `threading_layer`
  field.
  https://github.com/joblib/threadpoolctl/pull/48
  https://github.com/joblib/threadpoolctl/pull/60


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
