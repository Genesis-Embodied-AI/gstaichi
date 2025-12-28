#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/run_tests.py -v -r 3
