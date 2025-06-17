#!/bin/bash

set -ex

pip3 install -r requirements_test.txt
python -c 'import tests.test_utils; print("Available architectures", tests.test_utils.expected_archs())'
python3.10 tests/run_tests.py -v
