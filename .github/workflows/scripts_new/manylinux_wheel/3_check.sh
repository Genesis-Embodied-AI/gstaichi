#!/bin/bash

set -ex

pip install auditwheel
auditwheel show dist/*.whl
auditwheel repair -v dist/*.whl
auditwheel show dist/*.whl
