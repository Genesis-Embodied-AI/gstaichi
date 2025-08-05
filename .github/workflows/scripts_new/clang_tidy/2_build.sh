#!/bin/bash

set -ex

export GSTAICHI_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
./build.py wheel
