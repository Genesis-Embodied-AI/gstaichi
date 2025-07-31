#!/bin/bash

set -ex

pip install -r requirements_test.txt
python tests/run_tests.py -v -r 3
