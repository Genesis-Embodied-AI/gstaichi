#!/bin/bash

set -ex

# yum update
yum install -y git wget libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel
git config --global --add safe.directory /__w/taichi/taichi
git submodule update --init --jobs 2

wget https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.4/clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4.tar.xz
tar -xf clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4.tar.xz
