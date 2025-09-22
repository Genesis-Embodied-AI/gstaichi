# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .bootstrap import get_cache_home
from .cmake import cmake_args
from .dep import download_dep
from .misc import banner, get_cache_home


# -- code --
@banner("Setup LLVM")
def setup_llvm() -> None:
    """
    Download and install LLVM.
    """
    u = platform.uname()

    release_url_template = "https://github.com/Genesis-Embodied-AI/gstaichi-sdk-builds/releases/download/llvm-17.0.6-hp-llvm-build-no-arm-202509220138/taichi-llvm-17.0.6-{platform}.zip"
    out = get_cache_home() / "llvm17.0.6"

    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            out = f"{out}-amdgpu"
            url = "https://github.com/GaleSeLee/assets/releases/download/v0.0.5/taichi-llvm-15.0.0-linux.zip"
        else:
            url = release_url_template.format(platform="linux-x86_64")
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        url = release_url_template.format(platform="macos-arm64")
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        url = release_url_template.format(platform="windows-amd64")
        download_dep(url, out, strip=0)
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
