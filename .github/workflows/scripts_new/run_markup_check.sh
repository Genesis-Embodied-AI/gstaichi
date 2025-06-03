#!/bin/bash

set -ex
set -o pipefail

npm i --global -D markdown-link-check

find . -name \*.md -print0 | xargs -0 -n1 markdown-link-check -q -c .github/workflows/scripts_new/markdown-link-check-config.json 
