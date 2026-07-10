#!/bin/bash

set -xe

# step outside of threadpoolctl directory
pushd ..
ABS_PATH=$(pwd)

# build & install FlexiBLAS
mkdir flexiblas_install
git clone https://github.com/mpimd-csc/flexiblas.git
pushd flexiblas

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/ccache}"
mkdir -p "$CCACHE_DIR"
ccache --max-size=500M

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
FLEXIBLAS_LIB=$ABS_PATH/flexiblas_install/lib
cmake ../ -DCMAKE_INSTALL_PREFIX=$ABS_PATH"/flexiblas_install" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_INSTALL_RPATH=$FLEXIBLAS_LIB \
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_Fortran_COMPILER_LAUNCHER=ccache \
    -DBLAS_AUTO_DETECT="OFF" \
    -DEXTRA="OPENBLAS_CONDA" \
    -DFLEXIBLAS_DEFAULT="OPENBLAS_CONDA" \
    -DOPENBLAS_CONDA_LIBRARY=$CONDA_PREFIX"/lib/libopenblas"$EXTENSION \
make
make install

export LD_LIBRARY_PATH=$FLEXIBLAS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [[ $(uname) == "Darwin" ]]; then
    export DYLD_LIBRARY_PATH=$FLEXIBLAS_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}
fi

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

ccache -s || true

export CFLAGS=-I$ABS_PATH/flexiblas_install/include/flexiblas
export LDFLAGS="-L$FLEXIBLAS_LIB -Wl,-rpath,$FLEXIBLAS_LIB"

popd

# back to threadpoolctl directory
popd
