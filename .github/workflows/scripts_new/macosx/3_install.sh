#!/bin/bash

set -ex

pip install dist/*.whl
python -c "import taichi as ti; ti.init(arch=ti.cpu)"
python -c "import taichi as ti; ti.init(arch=ti.metal)"
