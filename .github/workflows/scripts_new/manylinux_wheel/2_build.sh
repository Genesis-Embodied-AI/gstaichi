#!/bin/bash

set -ex

export PATH=$PWD/clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4/bin:$PATH

export GSTAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=OFF"
./build.py wheel
