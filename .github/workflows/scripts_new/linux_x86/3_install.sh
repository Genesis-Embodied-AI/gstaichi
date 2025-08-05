#!/bin/bash

set -ex

pip3 install dist/*.whl
python -c "import gstaichi as ti; ti.init(arch=ti.cpu)"
