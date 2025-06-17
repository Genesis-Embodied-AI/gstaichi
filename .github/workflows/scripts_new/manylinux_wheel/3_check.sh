#!/bin/bash

set -ex

~/.cache/ti-build-cache/mambaforge/envs/3.12/bin/pip install auditwheel
auditwheel show dist/*.whl
