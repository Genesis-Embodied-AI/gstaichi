#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
python tests/run_tests.py -v -r 3
