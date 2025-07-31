#!/bin/bash

set -ex

export PATH=$PWD/clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4/bin:$PATH

export TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_BUILD_TESTS:BOOL=OFF"
./build.py wheel
