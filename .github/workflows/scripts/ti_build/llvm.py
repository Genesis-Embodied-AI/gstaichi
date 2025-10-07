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

    llvm_version = "16.0.6"
    build_version = "hp-llvm-u18-container-202509212058"
    release_url_template = "https://github.com/Genesis-Embodied-AI/gstaichi-sdk-builds/releases/download/llvm-{llvm_version}-{build_version}/taichi-llvm-{llvm_version}-{platform}.zip".format(
        llvm_version=llvm_version,
        build_version=build_version,
        platform="{platform}",
    )

    if u.system == "Linux":
        if cmake_args.get_effective("TI_WITH_AMDGPU"):
            raise Exception("AMD not currently supported")
        else:
            out = get_cache_home() / f"llvm-{llvm_version}-x86-{build_version}"
            url = release_url_template.format(platform="linux-x86_64")
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        out = get_cache_home() / f"llvm-{llvm_version}-{build_version}"
        url = release_url_template.format(platform="macos-arm64")
        download_dep(url, out, strip=1)
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        out = get_cache_home() / f"llvm-{llvm_version}-{build_version}"
        url = release_url_template.format(platform="windows-amd64")
        download_dep(url, out, strip=0)
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
