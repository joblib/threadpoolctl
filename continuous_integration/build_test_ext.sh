
cd threadpoolctl/tests/_openmp_test_helper
python setup.py build_ext -i || echo 'No openmp'
cd ../..