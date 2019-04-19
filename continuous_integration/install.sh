#!/bin/bash

set -e

UNAMESTR=`uname`

if [[ "$UNAMESTR" == "Darwin" ]]; then
    # Install a compiler with a working openmp
    HOMEBREW_NO_AUTO_UPDATE=1 brew install libomp

    # enable OpenMP support for Apple-clang
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
    export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
    export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
    export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

elif [[ "$CC_OUTER_LOOP" == "clang-8" || "$CC_INNER_LOOP" == "clang-8" ]]; then
    # Assume Ubuntu: install a recent version of clang and libomp
    echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
    echo "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" | sudo tee -a /etc/apt/sources.list.d/llvm.list
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    sudo apt update
    sudo apt install clang-8 libomp-8-dev
fi

sudo apt install gdb

make_conda() {
    TO_INSTALL="$@"
    conda create -n $VIRTUALENV -q --yes $TO_INSTALL
    source activate $VIRTUALENV
}

if [[ "$PACKAGER" == "conda" ]]; then
    TO_INSTALL="python=$VERSION_PYTHON pip pytest pytest-cov cython"
    if [[ "$NO_NUMPY" != "true" ]]; then
        TO_INSTALL="$TO_INSTALL numpy scipy"
        if [[ "$NO_MKL" == "true" ]]; then
            TO_INSTALL="$TO_INSTALL nomkl"
        fi
    fi
	make_conda $TO_INSTALL

elif [[ "$PACKAGER" == "ubuntu" ]]; then
    sudo apt-get install python3-scipy libatlas3-base libatlas-base-dev libatlas-dev libopenblas-base python3-virtualenv
    python3 -m virtualenv --system-site-packages --python=python3 $VIRTUALENV
    source $VIRTUALENV/bin/activate
fi


python -m pip install -q -r dev-requirements.txt
bash ./continuous_integration/build_test_ext.sh

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)" || echo "no numpy"
python -c "import scipy; print('scipy %s' % scipy.__version__)" || echo "no scipy"

python -m flit install --symlink
