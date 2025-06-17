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

pip install ruff
ruff check

# C++
# ===

# TODO: figure out how to run clang-tidy
