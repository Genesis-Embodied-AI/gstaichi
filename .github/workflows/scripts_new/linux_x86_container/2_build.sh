#!/bin/bash

set -ex

# ~/bin contains our fake 'sudo' command
export PATH=~/bin:$PATH
./build.py wheel
