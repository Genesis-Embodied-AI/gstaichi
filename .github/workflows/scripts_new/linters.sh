#!/bin/bash

set -ex

python -V
pwd
ls
uname -a

pip install pre-commit
pre-commit run -a --show-diff
