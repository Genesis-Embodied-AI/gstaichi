#!/bin/bash

set -ex

pip install auditwheel
auditwheel show dist/*.whl
auditwheel repair dist/*.whl
ls -lh /wheelhouse/
auditwheel show /wheelhouse/*.whl
