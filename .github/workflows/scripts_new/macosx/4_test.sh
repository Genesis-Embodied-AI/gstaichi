#!/bin/bash

set -ex

pip3.10 install -r requirements_test.txt
python3.10 tests/run_tests.py -v --cpp
python3.10 tests/run_tests.py -v -r 3
