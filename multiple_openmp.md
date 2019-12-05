# Multiple OpenMP runtimes

OpenMP is an API specification for parallel programming. There are many
implementations of it, tied to a compiler most of the time: the libgomp library
for GCC, libomp for LLVM/Clang, libiomp for ICC and vcomp for MSVC for example.
In the following, we refer to OpenMP without distinction between the
specification and an implementation.

In general, it's not advised to have different OpenMP libraries (or even
different copies of the same library) loaded at the same time in a program.
It's considered an undefined behavior. Fortunately it's not as bad as it sounds
in most situations.

However it can easily happen in the Python ecosystem since you can install
packages compiled with different compilers (hence linked to different OpenMP
implementations) and import them in a same Python program.

A typical example is installing numpy from Anaconda which ships MKL
(Intel's math library) and a package using multi-threading with OpenMP
(Scikit-learn, LightGBM, XGBoost, ...).

From our experience, most OpenMP libraries can seamlessly coexist in a same
program. For instance, on Linux, we never observed any issue between libgomp
and libiomp, which is the most common mix (numpy with MKL + a package compiled
with GCC, the most widely used C compiler).

The only unrecoverable incompatibility we encountered is between libomp
(LLVM/Clang) and libiomp (ICC), on Linux, manifested by crashes or deadlocks.
It can happen even with the simplest OpenMP calls like getting the maximum
number of threads that will be used in a subsequent parallel region. A possible
explanation is that libomp is actually a fork of libiomp causing name colliding
for instance. Using threadpoolctl may crash your program in such a setting.

Surprisingly, we never encountered this kind of issue on macOS, where this mix
is the most frequent (Clang being the default C compiler on macOS).

As far as we know, the only workaround consists in getting rid of one of the
OpenMP libraries. For example:

- Build your OpenMP-enabled extensions with GCC (or ICC) instead of Clang.

- Install a build of Numpy linked against OpenBLAS instead of MKL. This can be
  done for instance by installing Numpy from PyPI::

    pip install numpy

  from the conda-forge conda channel::

    conda install -c conda-forge numpy

  or from the default conda channel::

    conda install numpy blas[build=openblas]
