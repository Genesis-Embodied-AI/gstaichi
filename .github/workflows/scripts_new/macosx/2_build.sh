#!/bin/bash

set -ex

source .venv/bin/activate
export GSTAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_METAL:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"
./build.py wheel
