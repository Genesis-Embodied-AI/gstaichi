#!/bin/bash

set -ex

pip install --prefer-binary -r requirements_test.txt
python tests/run_tests.py -v -r 3
