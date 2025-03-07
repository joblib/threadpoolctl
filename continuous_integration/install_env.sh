#!/bin/bash

# License: BSD 3-Clause

set -xe

UNAMESTR=`uname`


# Install a recent version of clang and libomp if needed
# Only applicable on linux jobs
if [[ "$CC_OUTER_LOOP" == "clang-18" ]] || \
   [[ "$CC_INNER_LOOP" == "clang-18" ]] || \
   [[ "$BLIS_CC" == "clang-18" ]]
then
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh 18
    sudo apt-get install libomp-dev

    sudo find / -name "omp.h"
fi

# Install gcc 8 to build BLIS
if [[ "$BLIS_CC" == "gcc-8" ]]; then
    sudo apt install gcc-8
fi

make_conda() {
    CHANNEL="$1"
    TO_INSTALL="setuptools $2"
    if [[ "$UNAMESTR" == "Darwin" ]]; then
        if [[ "$INSTALL_LIBOMP" == "conda-forge" ]]; then
            # Install an OpenMP-enabled clang/llvm from conda-forge
            # assumes conda-forge is set on priority channel
            TO_INSTALL="$TO_INSTALL compilers llvm-openmp"

        elif [[ "$INSTALL_LIBOMP" == "homebrew" ]]; then
            # Install a compiler with a working openmp
            HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp

            # enable OpenMP support for Apple-clang
            export CC=/usr/bin/clang
            export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
            export CFLAGS="$CFLAGS -I/opt/homebrew/opt/libomp/include"
            export LDFLAGS="$LDFLAGS -Wl,-rpath,/opt/homebrew/opt/libomp/lib -L/opt/homebrew/opt/libomp/lib -lomp"
        fi
    fi

    if [[ "$PYTHON_VERSION" == "*" ]]; then
        # Avoid installing free-threaded python
        TO_INSTALL="$TO_INSTALL python-gil"
    fi

    conda config --add channels $CHANNEL
    conda update -n base conda conda-libmamba-solver -q --yes
    conda config --set solver libmamba
    conda create -n testenv -q --yes python=$PYTHON_VERSION $TO_INSTALL
    conda activate testenv

    which clang
}


if [[ "$PACKAGER" == "conda" ]]; then
    TO_INSTALL=""
    if [[ "$NO_NUMPY" != "true" ]]; then
        TO_INSTALL="$TO_INSTALL numpy scipy"
        if [[ -n "$BLAS" ]]; then
            TO_INSTALL="$TO_INSTALL blas=*=$BLAS"
        fi
    fi
	make_conda "defaults" "$TO_INSTALL"

elif [[ "$PACKAGER" == "conda-forge" ]]; then
    TO_INSTALL="numpy scipy blas=*=$BLAS"
    if [[ "$BLAS" == "openblas" && "$OPENBLAS_THREADING_LAYER" == "openmp" ]]; then
        TO_INSTALL="$TO_INSTALL libopenblas=*=*openmp*"
    fi
    make_conda "conda-forge" "$TO_INSTALL"

elif [[ "$PACKAGER" == "pip" ]]; then
    # Use conda to build an empty python env and then use pip to install
    # numpy and scipy
    make_conda "conda-forge" ""
    if [[ "$NO_NUMPY" != "true" ]]; then
        pip install numpy scipy
    fi

elif [[ "$PACKAGER" == "pip-dev" ]]; then
    # Use conda to build an empty python env and then use pip to install
    # numpy and scipy dev versions
    make_conda "conda-forge" ""

    dev_anaconda_url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple
    pip install --pre --upgrade --timeout=60 --extra-index $dev_anaconda_url numpy scipy

elif [[ "$PACKAGER" == "ubuntu" ]]; then
    # Remove the ubuntu toolchain PPA that seems to be invalid:
    # https://github.com/scikit-learn/scikit-learn/pull/13934
    sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install python3-scipy python3-virtualenv $APT_BLAS
    python3 -m virtualenv --system-site-packages --python=python3 testenv
    source testenv/bin/activate

elif [[ "$INSTALL_BLAS" == "BLIS" ]]; then
    TO_INSTALL="cython meson-python pkg-config"
    make_conda "conda-forge" "$TO_INSTALL"
    bash ./continuous_integration/install_blis.sh

elif [[ "$INSTALL_BLAS" == "FlexiBLAS" ]]; then
    TO_INSTALL="cython openblas $PLATFORM_SPECIFIC_PACKAGES meson-python pkg-config compilers"
    make_conda "conda-forge" "$TO_INSTALL"
    bash ./continuous_integration/install_flexiblas.sh

fi


python -m pip install -v -q -r dev-requirements.txt
bash ./continuous_integration/build_test_ext.sh

# Check which BLAS is linked
ldd tests/_openmp_test_helper/nested_prange_blas.cpython*.so

python --version
python -c "import numpy; print(f'numpy {numpy.__version__}')" || echo "no numpy"
python -c "import scipy; print(f'scipy {scipy.__version__}')" || echo "no scipy"

python -m flit install --symlink
