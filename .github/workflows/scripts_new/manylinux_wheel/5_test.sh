#!/bin/bash

set -ex

pip install --group test
python tests/run_tests.py -v -r 3
