#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
python tests/run_tests.py -v -r 3
