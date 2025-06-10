#!/bin/bash

set -ex

python python/tools/run_modified_files.py --include 'python/*.py' -- pyright
