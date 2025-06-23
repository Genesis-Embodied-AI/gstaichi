#!/bin/bash

set -ex

pip3.10 install dist/*.whl
python -c "import taichi as ti; ti.init(arch=ti.cpu)"
python -c "import taichi as ti; ti.init(arch=ti.metal)"
