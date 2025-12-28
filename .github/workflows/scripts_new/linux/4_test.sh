#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
export TI_LIB_DIR="$(python -c 'import gstaichi as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/gstaichi_cpp_tests  --gtest_filter=-AMDGPU.*
python tests/run_tests.py -v -r 3
