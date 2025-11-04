#!/bin/bash

set -ex

pwd
uname -a
git status
free -m
cat /etc/lsb-release
ls -la
python -V

LLVM_DIR=$(python download_llvm.py | tail -n 1)
export PATH=${LLVM_DIR}/bin:$PATH
chmod +x ${LLVM_DIR}/bin/*
clang --version
which clang

python -c 'import sys; print("sys.platform", sys.platform)'
python -c 'import os; print("os.uname()[4]", os.uname()[4])'

git submodule
git submodule update --init --recursive
sudo apt update
sudo apt install -y \
    pybind11-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    libclang-common-15-dev \
    cmake \
    ninja-build

pip3 install scikit-build
