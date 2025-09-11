#!/bin/bash

set -ex

pip install --prefer-binary --group test
find . -name '*.bc'
ls -lh build/
export TI_LIB_DIR="$(python -c 'import gstaichi as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/gstaichi_cpp_tests
python tests/run_tests.py -v -r 3
