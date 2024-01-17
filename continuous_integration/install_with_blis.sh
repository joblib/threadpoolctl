#!/bin/bash

set -e

pushd ..
ABS_PATH=$(pwd)
popd

# Install a recent version of clang and libomp
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 10
sudo apt-get install libomp-dev

# create conda env
conda update -n base conda conda-libmamba-solver -q --yes
conda config --set solver libmamba
conda create -n $VIRTUALENV -q --yes -c conda-forge python=$PYTHON_VERSION \
    pip cython meson-python pkg-config
source activate $VIRTUALENV

if [[ "$BLIS_CC" == "gcc-8" ]]; then
    sudo apt install gcc-8
fi

pushd ..

# build & install blis
mkdir BLIS_install
git clone https://github.com/flame/blis.git
pushd blis

./configure --prefix=$ABS_PATH/BLIS_install --enable-cblas --enable-threading=$BLIS_ENABLE_THREADING CC=$BLIS_CC auto
make -j4
make install
popd

# build & install numpy
git clone https://github.com/numpy/numpy.git
pushd numpy
git submodule update --init

echo "libdir=$ABS_PATH/BLIS_install/lib/
includedir=$ABS_PATH/BLIS_install/include/blis/
version=latest
extralib=-lm -lpthread -lgfortran
Name: blis
Description: BLIS
Version: \${version}
Libs: -L\${libdir} -lblis
Libs.private: \${extralib}
Cflags: -I\${includedir}" > blis.pc

PKG_CONFIG_PATH=$ABS_PATH/numpy/ pip install . -v --no-build-isolation -Csetup-args=-Dblas=blis
popd

popd

python -m pip install -q -r dev-requirements.txt
CFLAGS=-I$ABS_PATH/BLIS_install/include/blis \
    LDFLAGS="-L$ABS_PATH/BLIS_install/lib -Wl,-rpath,$ABS_PATH/BLIS_install/lib" \
    bash ./continuous_integration/build_test_ext.sh

# Check that BLIS is linked
ldd tests/_openmp_test_helper/nested_prange_blas.cpython*.so

python --version
python -c "import numpy; print(f'numpy {numpy.__version__}')" || echo "no numpy"
python -c "import scipy; print(f'scipy {scipy.__version__}')" || echo "no scipy"

python -m flit install --symlink
