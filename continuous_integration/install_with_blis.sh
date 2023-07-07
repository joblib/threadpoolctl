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

## create conda env from numpy requirements
conda create -n $VIRTUALENV python=3.9 pip cython setuptools=59.2.0 meson ninja pkg-config meson-python
source activate $VIRTUALENV

git submodule update --init
echo "[blis]
libraries = blis
library_dirs = $ABS_PATH/BLIS_install/lib
include_dirs = $ABS_PATH/BLIS_install/include/blis
runtime_library_dirs = $ABS_PATH/BLIS_install/lib" > site.cfg
python setup.py develop
popd

popd

python -m pip install -q -r dev-requirements.txt
CFLAGS=-I$ABS_PATH/BLIS_install/include/blis LDFLAGS=-L$ABS_PATH/BLIS_install/lib \
    bash ./continuous_integration/build_test_ext.sh

python --version
python -c "import numpy; print(f'numpy {numpy.__version__}')" || echo "no numpy"
python -c "import scipy; print(f'scipy {scipy.__version__}')" || echo "no scipy"

python -m flit install --symlink
