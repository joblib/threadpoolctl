#!/bin/bash

set -e

pushd ..
ABS_PATH=$(pwd)
popd

# create conda env
conda update -n base conda conda-libmamba-solver -q --yes
conda config --set solver libmamba
conda create -n $VIRTUALENV -q --yes -c conda-forge python=$PYTHON_VERSION \
    pip cython blis mkl openblas
source activate $VIRTUALENV

pushd ..

# build & install flexiblas
mkdir flexiblas_install
git clone https://github.com/mpimd-csc/flexiblas.git
pushd flexiblas

mkdir build
pushd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$ABS_PATH"/flexiblas_install" \
    -DEXTRA="OpenBLAS;BLIS;MKL" \
    -DOpenBLAS_LIBRARY=$CONDA_PREFIX"/lib/libopenblas.so" \
    -DBLIS_LIBRARY=$CONDA_PREFIX"/lib/libblis.so" \
    -DMKL_LIBRARY=$CONDA_PREFIX"/lib/libmkl_rt.so"
make
make install
popd
popd

popd

python -m pip install -q -r dev-requirements.txt
CFLAGS=-I$ABS_PATH/flexiblas_install/include/flexiblas \
    LDFLAGS="-L$ABS_PATH/flexiblas_install/lib -Wl,-rpath,$ABS_PATH/flexiblas_install/lib" \
    bash ./continuous_integration/build_test_ext.sh

# Check that flexiblas is linked
ldd tests/_openmp_test_helper/nested_prange_blas.cpython*.so

python --version
python -c "import numpy; print(f'numpy {numpy.__version__}')" || echo "no numpy"
python -c "import scipy; print(f'scipy {scipy.__version__}')" || echo "no scipy"

python -m flit install --symlink