#!/bin/bash

set -ex

python ./scripts/run_clang_tidy.py $PWD/gs_taichi -clang-tidy-binary clang-tidy-14 -header-filter=$PWD/taichi -j2
