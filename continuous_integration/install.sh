#!/bin/bash

set -e

UNAMESTR=`uname`

if [[ "$CC_OUTER_LOOP" == "clang-8" || "$CC_INNER_LOOP" == "clang-8" ]]; then
    # Assume Ubuntu: install a recent version of clang and libomp
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
    echo "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt update
    sudo apt install clang-8 libomp-8-dev
fi

make_conda() {
    TO_INSTALL="$@"
    if [[ "$UNAMESTR" == "Darwin" ]]; then
        if [[ "$INSTALL_LIBOMP" == "conda-forge" ]]; then
            # Install an OpenMP-enabled clang/llvm from conda-forge
            TO_INSTALL="$TO_INSTALL conda-forge::compilers conda-forge::llvm-openmp"
            export CFLAGS="$CFLAGS -I$CONDA/envs/$VIRTUALENV/include"
            export LDFLAGS="$LDFLAGS -Wl,-rpath,$CONDA/envs/$VIRTUALENV/lib -L$CONDA/envs/$VIRTUALENV/lib"

        elif [[ "$INSTALL_LIBOMP" == "homebrew" ]]; then
            # Install a compiler with a working openmp
            HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp

            # enable OpenMP support for Apple-clang
            export CC=/usr/bin/clang
            export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
            export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
            export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
        fi
    fi
    conda create -n $VIRTUALENV -q --yes $TO_INSTALL
    source activate $VIRTUALENV
}

if [[ "$PACKAGER" == "conda" ]]; then
    TO_INSTALL="python=$VERSION_PYTHON pip"
    if [[ "$NO_NUMPY" != "true" ]]; then
        TO_INSTALL="$TO_INSTALL numpy scipy blas[build=$BLAS]"
    fi
	make_conda $TO_INSTALL

elif [[ "$PACKAGER" == "conda-forge" ]]; then
    conda config --prepend channels conda-forge
    conda config --set channel_priority strict
    TO_INSTALL="python=$VERSION_PYTHON numpy scipy"
    make_conda $TO_INSTALL

elif [[ "$PACKAGER" == "pip" ]]; then
    # Use conda to build an empty python env and then use pip to install
    # numpy and scipy
    TO_INSTALL="python=$VERSION_PYTHON pip"
    make_conda $TO_INSTALL
    if [[ "$NO_NUMPY" != "true" ]]; then
        pip install numpy scipy
    fi

elif [[ "$PACKAGER" == "ubuntu" ]]; then
    # Remove the ubuntu toolchain PPA that seems to be invalid:
    # https://github.com/scikit-learn/scikit-learn/pull/13934
    sudo add-apt-repository --remove ppa:ubuntu-toolchain-r/test
    sudo apt-get update
    sudo apt-get install python3-scipy python3-virtualenv $APT_BLAS
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
fi


python -m pip install -q -r dev-requirements.txt
bash ./continuous_integration/build_test_ext.sh

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)" || echo "no numpy"
python -c "import scipy; print('scipy %s' % scipy.__version__)" || echo "no scipy"

python -m flit install --symlink
