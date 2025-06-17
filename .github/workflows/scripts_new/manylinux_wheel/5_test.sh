#!/bin/bash

set -ex

ls -lR 

pip install -r requirements_test.txt
python tests/run_tests.py
