#!/bin/bash

set -ex

pybind11-stubgen taichi._lib.core.taichi_python --ignore-all-errors
