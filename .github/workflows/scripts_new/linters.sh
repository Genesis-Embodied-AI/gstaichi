#!/bin/bash

set -ex

python -V
pwd
ls
uname -a

# python + C++
# =============

pip install pre-commit
pre-commit run -a --show-diff

# python
# ======

pip install pyright
# Need to deal with C++ linkage issues first (and possibly
# some other things), before we can turn on pyright

pip install isort
# TODO: run isort on all python files, and commit those, then
# uncomment the following line:
# isort --check-only --diff python

# C++
# ===

# TODO: figure out how to run clang-tidy
