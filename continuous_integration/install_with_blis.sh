#!/bin/bash

set -e

pushd ..
ABS_PATH=$(pwd)
popd

# Assume Ubuntu: install a recent version of clang and libomp
echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
echo "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt update
sudo apt install clang-8 libomp-8-dev

# create conda env
conda create -n $VIRTUALENV -q --yes python=$VERSION_PYTHON pip cython
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
echo "[blis]
libraries = blis
library_dirs = $ABS_PATH/BLIS_install/lib
include_dirs = $ABS_PATH/BLIS_install/include/blis
runtime_library_dirs = $ABS_PATH/BLIS_install/lib" > site.cfg
python setup.py build_ext -i
pip install -e .
popd

popd

python -m pip install -q -r dev-requirements.txt
CFLAGS=-I$ABS_PATH/BLIS_install/include/blis LDFLAGS=-L$ABS_PATH/BLIS_install/lib \
    bash ./continuous_integration/build_test_ext.sh

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)" || echo "no numpy"
python -c "import scipy; print('scipy %s' % scipy.__version__)" || echo "no scipy"

python -m flit install --symlink
