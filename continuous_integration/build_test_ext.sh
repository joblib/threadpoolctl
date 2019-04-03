#!/bin/bash

set -e

pushd tests/_openmp_test_helper
rm -rf *.c *.so *.dylib build/
python setup_inner.py build_ext -i
python setup_outer.py build_ext -i
popd

