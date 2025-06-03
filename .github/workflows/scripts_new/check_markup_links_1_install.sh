#!/bin/bash

set -ex

curl -sSf 'https://sh.rustup.rs' | sh
sudo apt-get install -y gcc pkg-config libc6-dev libssl-dev
