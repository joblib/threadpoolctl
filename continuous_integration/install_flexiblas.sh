#!/bin/bash

set -xe

# step outside of threadpoolctl directory
pushd ..
ABS_PATH=$(pwd)

# build & install FlexiBLAS
mkdir flexiblas_install
git clone https://github.com/mpimd-csc/flexiblas.git
pushd flexiblas

# Temporary ping Flexiblas commit to avoid openmp symbols not found at link time
git checkout v3.5.0

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
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
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

export CFLAGS=-I$ABS_PATH/flexiblas_install/include/flexiblas \
export LDFLAGS="-L$ABS_PATH/flexiblas_install/lib -Wl,-rpath,$ABS_PATH/flexiblas_install/lib" \

popd

# back to threadpoolctl directory
popd
