#!/bin/bash

set -ex

PLATFORM=$(uname -m)
if [ "$PLATFORM" = "x86_64" ]; then
    PLATFORM="x86_64"
elif [ "$PLATFORM" = "aarch64" ]; then
    PLATFORM="aarch64"
else
    echo "Unsupported architecture: $PLATFORM"
    exit 1
fi

echo "Detected platform: $PLATFORM"

export PATH=$PWD/taichi-llvm-15.0.7-linux-${PLATFORM}/bin:$PATH

export GSTAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"
./build.py wheel
