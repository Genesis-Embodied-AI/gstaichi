#!/bin/bash

set -ex

CHANGED_FILES=$(git diff --name-only --diff-filter=d origin/main...HEAD | grep '\.py$')

if [ -n "$CHANGED_FILES" ]; then
    echo "$CHANGED_FILES" | xargs pyright
else
    echo "No Python files changed"
fi
