#!/bin/bash

set -ex

ls -lR manylinux_wheel

python -V
pip -V

pip install manylinux_wheel/dist/*.whl

python -c 'import taichi as ti; ti.init(arch=ti.cpu); print(ti.__version__)'
