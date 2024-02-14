#!/bin/bash

set -e

pushd ..
ABS_PATH=$(pwd)
popd

# create conda env
conda update -n base conda conda-libmamba-solver -q --yes
conda config --set solver libmamba
conda create -n $VIRTUALENV -q --yes -c conda-forge python=$PYTHON_VERSION \
    pip cython openblas $PLATFORM_SPECIFIC_PACKAGES meson-python pkg-config compilers
source activate $VIRTUALENV

pushd ..

# build & install FlexiBLAS
mkdir flexiblas_install
git clone https://github.com/mpimd-csc/flexiblas.git
pushd flexiblas

mkdir build
pushd build

EXTENSION=".so"
if [[ $(uname) == "Darwin" ]]; then
    EXTENSION=".dylib"
fi

# We intentionally restrict the list of backends to make it easier to
# write platform agnostic tests. In particular, we do not detect OS
# provided backends such as macOS' Apple/Accelerate/vecLib nor plaftorm
# specific BLAS implementations such as MKL that cannot be installed on
# arm64 hardware.
cmake ../ -DCMAKE_INSTALL_PREFIX=$ABS_PATH"/flexiblas_install" \
    -DBLAS_AUTO_DETECT="OFF" \
    -DEXTRA="OPENBLAS_CONDA" \
    -DFLEXIBLAS_DEFAULT="OPENBLAS_CONDA" \
    -DOPENBLAS_CONDA_LIBRARY=$CONDA_PREFIX"/lib/libopenblas"$EXTENSION \
make
make install

# Check that all 3 BLAS are listed in FlexiBLAS configuration
$ABS_PATH/flexiblas_install/bin/flexiblas list
popd
popd

# build & install numpy
git clone https://github.com/numpy/numpy.git
pushd numpy
git submodule update --init

echo "libdir=$ABS_PATH/flexiblas_install/lib/
includedir=$ABS_PATH/flexiblas_install/include/flexiblas/
version=3.3.1
extralib=-lm -lpthread -lgfortran
Name: flexiblas
Description: FlexiBLAS - a BLAS wrapper
Version: \${version}
Libs: -L\${libdir} -lflexiblas
Libs.private: \${extralib}
Cflags: -I\${includedir}" > flexiblas.pc

PKG_CONFIG_PATH=$ABS_PATH/numpy/ pip install . -v --no-build-isolation -Csetup-args=-Dblas=flexiblas -Csetup-args=-Dlapack=flexiblas
popd

popd

python -m pip install -q -r dev-requirements.txt
CFLAGS=-I$ABS_PATH/flexiblas_install/include/flexiblas \
    LDFLAGS="-L$ABS_PATH/flexiblas_install/lib -Wl,-rpath,$ABS_PATH/flexiblas_install/lib" \
    bash ./continuous_integration/build_test_ext.sh

# Check that FlexiBLAS is linked
if [[ $(uname) != "Darwin" ]]; then
    ldd tests/_openmp_test_helper/nested_prange_blas.cpython*.so
fi

python --version
python -c "import numpy; print(f'numpy {numpy.__version__}')" || echo "no numpy"
python -c "import scipy; print(f'scipy {scipy.__version__}')" || echo "no scipy"

python -m flit install --symlink
