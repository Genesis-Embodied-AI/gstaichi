#!/bin/bash

set -ex

pip install -r requirements_test.txt
TI_LIB_DIR=$(python -c "import os; import gstaichi as ti; p = os.path.join(ti.__path__[0], '_lib', 'runtime'); print(p)" | tail -n 1)
export TI_LIB_DIR
./build/gstaichi_cpp_tests
python tests/run_tests.py -v -r 3
