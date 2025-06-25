#!/bin/bash

set -ex

pip install auditwheel
auditwheel show dist/*.whl
