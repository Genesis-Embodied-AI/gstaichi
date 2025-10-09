# -*- coding: utf-8 -*-

# -- stdlib --
import os
import platform

# -- third party --
# -- own --
from .bootstrap import get_cache_home
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
    build_version = "202510071838"
    release_url_template = "https://github.com/Genesis-Embodied-AI/gstaichi-sdk-builds/releases/download/llvm-{llvm_version}-{build_version}/taichi-llvm-{llvm_version}-{platform}.zip".format(
        llvm_version=llvm_version,
        build_version=build_version,
        platform="{platform}",
    )
    out = get_cache_home() / f"llvm-{llvm_version}-{build_version}"

    strip = 1
    if u.system == "Linux":
        target_platform = "linux-x86_64"
    elif (u.system, u.machine) == ("Darwin", "arm64"):
        target_platform = "macos-arm64"
    elif (u.system, u.machine) == ("Windows", "AMD64"):
        target_platform = "windows-amd64"
        strip = 0
    else:
        raise RuntimeError(f"Unsupported platform: {u.system} {u.machine}")

    url = release_url_template.format(platform=target_platform)
    download_dep(url, out, strip=strip)

    # We should use LLVM toolchains shipped with OS.
    # path_prepend('PATH', out / 'bin')
    os.environ["LLVM_DIR"] = str(out)
