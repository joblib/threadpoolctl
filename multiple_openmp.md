# Multiple OpenMP Runtimes

## Context

OpenMP is an API specification for parallel programming. There are many
implementations of it, tied to a compiler most of the time:

-  `libgomp` for GCC (GNU C/C++ Compiler),
-  `libomp` for Clang (LLVM C/C++ Compiler),
-  `libiomp` for ICC (Intel C/C++ Compiler),
-  `vcomp` for MSVC (Microsoft Visual Studio C/C++ Compiler).

In general, it is not advised to have different OpenMP runtime libraries (or
even different copies of the same library) loaded at the same time in a
program. It's considered an undefined behavior. Fortunately it is not as bad as
it sounds in most situations.

However this situation is frequent in the Python ecosystem since you can
install packages compiled with different compilers (hence linked to different
OpenMP implementations) and import them together in a Python program.

A typical example is installing NumPy from Anaconda which is linked against MKL
(Intel's math library) and another package that uses multi-threading with OpenMP
directly in a compiled extension, as is the case in Scikit-learn (via Cython
`prange`), LightGBM and XGBoost (via pragmas in the C++ source code).

From our experience, **most OpenMP libraries can seamlessly coexist in a same
program**. For instance, on Linux, we never observed any issue between
`libgomp` and `libiomp`, which is the most common mix (NumPy with MKL + a
package compiled with GCC, the most widely used C compiler on that platform).

## Incompatibility between Intel OpenMP and LLVM OpenMP

The only unrecoverable incompatibility we encountered happens when loading a
mix of compiled extensions linked with **`libomp` (LLVM/Clang) and `libiomp`
(ICC), on Linux and macOS**, manifested by crashes or deadlocks. It can happen
even with the simplest OpenMP calls like getting the maximum number of threads
that will be used in a subsequent parallel region. A possible explanation is that
`libomp` is actually a fork of `libiomp` causing name colliding for instance.
Using `threadpoolctl` may crash your program in such a setting.

**Fortunately this problem is very rare**: at the time of writing, all major
binary distributions of Python packages for Linux use either GCC or ICC to
build the Python scientific packages. Therefore this problem would only happen
if some packagers decide to start shipping Python packages built with
LLVM/Clang instead of GCC (this is the case for instance with conda's default channel).

## Workarounds for Intel OpenMP and LLVM OpenMP case

As far as we know, the only workaround consists in making sure only of one of
the two incompatible OpenMP libraries is loaded. For example:

- Tell MKL (used by NumPy) to use another threading implementation instead of the Intel
  OpenMP runtime. It can be the GNU OpenMP runtime on Linux or TBB on Linux and MacOS
  for instance. This is done by setting the following environment variable:

      export MKL_THREADING_LAYER=GNU

  or, if TBB is installed:

      export MKL_THREADING_LAYER=TBB

- Install a build of NumPy and SciPy linked against OpenBLAS instead of MKL.
  This can be done for instance by installing NumPy and SciPy from PyPI:

      pip install numpy scipy

  from the conda-forge conda channel:

      conda install -c conda-forge numpy scipy

  or from the default conda channel:

      conda install numpy scipy blas[build=openblas]

- Re-build your OpenMP-enabled extensions from source with GCC (or ICC) instead
  of Clang if you want to keep on using NumPy/SciPy linked against MKL with the
  default `libiomp`-based threading layer.

## References

The above incompatibility has been reported upstream to the LLVM and Intel
developers on the following public issue trackers/forums along with a minimal
reproducer written in C:

- https://bugs.llvm.org/show_bug.cgi?id=43565
- https://community.intel.com/t5/Intel-C-Compiler/Cannot-call-OpenMP-functions-from-libiomp-after-calling-from/m-p/1176406
