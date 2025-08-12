#!/bin/bash

set -ex

pip install --prefer-binary -r requirements_test.txt
find . -name '*.bc'
ls -lh build/
ls -lh python/gstaichi/_lib/runtime/
export TI_LIB_DIR=python/gstaichi/_lib/runtime
./build/gstaichi_cpp_tests
python tests/run_tests.py -v -r 3
