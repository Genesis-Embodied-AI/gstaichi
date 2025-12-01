#!/bin/bash

set -ex

source .venv/bin/activate
export GSTAICHI_CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
./build.py wheel
