#!/bin/bash

set -ex

# yum update
yum install -y git wget
# Note: following depends on the name of the repo:
git config --global --add safe.directory /__w/gstaichi/gstaichi
git submodule update --init --jobs 2

# wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.4/clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4.tar.xz
# tar -xf clang+llvm-15.0.4-x86_64-linux-gnu-rhel-8.4.tar.xz

wget -q https://github.com/Genesis-Embodied-AI/gstaichi-sdk-builds/releases/download/llvm-15.0.7-202510071403/taichi-llvm-15.0.7-linux-x86_64.zip
unzip taichi-llvm-15.0.7-linux-x86_64.zip
ls
mkdirs -p ~/.cache/ti-build-cache
# yes, this is super hacky...
mv taichi-llvm-15.0.7-linux-x86_64 ~/.cache/ti-build-cache/llvm-15.0.7-202510071403
ls ~/.cache/ti-build-cache/
df -h

# clang++ searches for libstd++.so, not libstdc++.so.6
# without this, then the compiler checks will fail
# eg:
# - check for working compiler itself
# - and also check for -Wno-unused-but-set-variable, in GsTaichiCXXFlags.cmake
#   which will cause obscure compile errors for external/Eigen
ln -s /usr/lib64/libstdc++.so.6 /usr/lib64/libstdc++.so

# since we are linking statically
# and looks like this installs the same version of libstdc++-static as libstdc++
yum install -y libstdc++-static
