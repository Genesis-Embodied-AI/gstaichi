#!/bin/bash

set -ex

pwd
uname -a
git status
free -m
cat /etc/lsb-release
ls -la
python -V

git submodule
git submodule update --init --recursive
sudo apt update
sudo apt install -y \
    pybind11-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    clang-15 \
    libclang-common-15-dev \
    libclang-cpp15 \
    libclang1-15 \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip

pip3 install scikit-build
