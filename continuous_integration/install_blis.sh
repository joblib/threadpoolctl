#!/bin/bash

set -xe

# step outside of threadpoolctl directory
pushd ..
ABS_PATH=$(pwd)

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

export CFLAGS=-I$ABS_PATH/BLIS_install/include/blis
export LDFLAGS="-L$ABS_PATH/BLIS_install/lib -Wl,-rpath,$ABS_PATH/BLIS_install/lib"

popd

# back to threadpoolctl directory
popd
