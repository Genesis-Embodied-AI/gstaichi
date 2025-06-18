#!/bin/bash

set -ex

ls -l
find . -cmin -1
find . -cmin -1 -name '*.whl
find . -name '*.whl

python -V
pip -V

pip install manylinux_wheel/dist/*.whl

python -c 'import taichi as ti; ti.init(arch=ti.cpu); print(ti.__version__)'
