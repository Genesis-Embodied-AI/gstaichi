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

pip install mypy
# FIXME: Need to deal with all the ti.template() and similar, which
# violate typing/myp rules. And then we can run:
# mypy python

pip install isort
# TODO: run isort on all python files, and commit those, then
# uncomment the following line:
# isort --check-only --diff python

# C++
# ===

# TODO: figure out how to run clang-tidy
