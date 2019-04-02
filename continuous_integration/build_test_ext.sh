
pushd tests/_openmp_test_helper
rm *.so
python setup.py build_ext -i
popd

pushd tests/_openmp_test_helper_outer
rm *.so
python setup.py build_ext -i
popd
