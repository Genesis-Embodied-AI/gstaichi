#!/bin/bash

set -ex

pip install --prefer-binary --group test
pip install -r requirements_test_xdist.txt
find . -name '*.bc'
ls -lh build/
export TI_LIB_DIR="$(python -c 'import gstaichi as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
chmod +x ./build/gstaichi_cpp_tests
./build/gstaichi_cpp_tests
python tests/run_tests.py -v -r 3 --arch metal,vulkan,cpu
