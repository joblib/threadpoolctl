
pushd tests/_openmp_test_helper
rm -f *.so *.dylib
python setup.py build_ext -i
popd

pushd tests/_openmp_test_helper_outer
rm -f *.so *.dylib
python setup.py build_ext -i
popd
