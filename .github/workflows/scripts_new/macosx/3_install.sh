#!/bin/bash

set -ex

pip install dist/*.whl
python -c "import gstaichi as ti; ti.init(arch=ti.cpu)"
python -c "import gstaichi as ti; ti.init(arch=ti.metal)"
find . -name '*.bc'
