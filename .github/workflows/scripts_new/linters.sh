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
# since runnign pyright on everything is chaos right now,
# we only run it on modified files
# we factorize this to a separate script, so people can run it
# directly if they wish
bash .github/workflows/scripts_new/run_pyright_changed_files.sh

pip install isort
# TODO: run isort on all python files, and commit those, then
# uncomment the following line:
# isort --check-only --diff python

# C++
# ===

# TODO: figure out how to run clang-tidy
