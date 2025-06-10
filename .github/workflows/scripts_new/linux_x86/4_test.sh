#!/bin/bash

set -ex

pip3 install -r requirements_test.txt
python3.10 tests/run_tests.py -v -t 1
