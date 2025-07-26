#!/bin/bash

set -ex

pwd

python --version
sw_vers
clang++ --version
uname -a
clang --version
ls -la
python -V

pip install scikit-build
pip install numpy

brew install llvm@15

git submodule
git submodule update --init --recursive

brew install pybind11
which python
which python3
which pip
