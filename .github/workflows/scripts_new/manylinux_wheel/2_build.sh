#!/bin/bash

set -ex

export PATH=$PWD/clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4/bin:$PATH

./build.py wheel
