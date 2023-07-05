#!/bin/bash

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    pushd tests/_pyMylib
    rm -rf *.so *.o
    gcc -c -Wall -Werror -fpic -o my_threaded_lib.o my_threaded_lib.c
    gcc -shared -o my_threaded_lib.so my_threaded_lib.o
    popd
fi

pushd tests/_openmp_test_helper
rm -rf *.c *.so *.dylib build/
python setup_inner.py build_ext -i
python setup_outer.py build_ext -i

# skip scipy required extension if no numpy
if [[ "$NO_NUMPY" != "true" ]]; then
    python setup_nested_prange_blas.py build_ext -i
fi
popd
