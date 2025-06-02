#!/bin/bash

set -ex

echo hello from linux_x86.sh
pwd
uname -a
git status
git submodule
git submodule update --init --recursive
free -m
cat /etc/lsb-release
ls -la
python -V
sudo apt update
sudo apt install -y \
    freeglut3-dev \
    libglfw3-dev \
    libglm-dev \
    libglu1-mesa-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxcb-dri3-dev \
    libxcb-ewmh-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
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

./build.py wheel

pip3 install dist/*.whl
python -c "import taichi as ti; ti.init(arch=ti.cpu)"

pip3 install -r requirements_test.txt
python3.10 tests/run_tests.py -v
