#!/bin/bash

set -ex

pwd
dpkg -l '*cuda*' | grep 'ii'
uname -a
whoami
ls
apt-get update
apt-get install -y git wget python3 python3-pip
ln -s /usr/bin/python3 /usr/bin/python
python -V
pip --version
git status
git submodule
git submodule update --init --recursive
free -m
cat /etc/lsb-release
ls -la
python -V
apt-get install -y \
    freeglut3-dev \
    libglfw3-dev \
    libglm-dev \
    libglu1-mesa-dev \
    libwayland-dev \
    libx11-xcb-dev \
    libxcb-dri3-dev \
    libxcb-ewmh-dev \
    libxcb-keysyms1-dev \
    libxcb-randr0-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxrandr-dev \
    pybind11-dev \
    libc++-15-dev \
    libc++abi-15-dev \
    clang-15 \
    libclang-common-15-dev \
    libclang-cpp15 \
    libclang1-15 \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip

pip3 install scikit-build

apt-get install libjpeg-dev liblz4-dev libpng-dev libssl-dev libzstd-dev -y

# create fake 'sudo' script
mkdir -p ~/bin
cat >~/bin/sudo <<EOF
#!/bin/sh
exec "\$@"
exit 0
EOF
chmod +x ~/bin/sudo
