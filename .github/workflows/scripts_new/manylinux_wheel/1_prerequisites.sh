#!/bin/bash

set -ex

yum update
yum install -y git clang libXrandr-devel libXinerama-devel libXcursor-devel libXi-devel
git submodule update --init --jobs 2

