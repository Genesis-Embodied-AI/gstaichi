#!/bin/bash

set -ex

echo hello from macos_15.sh
pwd
git submodule
git submodule update --init --recursive
brew list
brew install python@3.10
brew install pybind11
brew link --force python@3.10
which python
which python3
sudo rm /opt/homebrew/bin/python3
sudo rm /Library/Frameworks/Python.framework/Versions/Current/bin/python
sudo ln -s /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3
sudo ln -s /opt/homebrew/bin/python3.10 /Library/Frameworks/Python.framework/Versions/Current/bin/python
python --version
sw_vers
clang++ --version
uname -a
clang --version
ls -la
python -V
pip3.10 install scikit-build
pip3.10 install numpy
brew install llvm@15
export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=OFF -DTI_WITH_METAL:BOOL=ON"
./build.py wheel

pip3.10 install dist/*.whl
python -c "import taichi as ti; ti.init(arch=ti.cpu)"
python -c "import taichi as ti; ti.init(arch=ti.metal)"

pip3.10 install -r requirements_test.txt
python3.10 tests/run_tests.py -v
