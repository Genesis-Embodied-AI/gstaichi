#!/bin/bash

set -ex

find . -cmin -1 -name '*.whl
find . -name '*.whl
ls -lh *.whl

python -V
pip -V

pip install dist/*.whl

python -c 'import taichi as ti; ti.init(arch=ti.cpu); print(ti.__version__)'
