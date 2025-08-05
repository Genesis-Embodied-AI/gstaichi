#!/bin/bash

set -ex

pip install -r requirements_test.txt
export TI_LIB_DIR=python/gstaichi/_lib/runtime
./build/gstaichi_cpp_tests
python tests/run_tests.py -v -r 3
