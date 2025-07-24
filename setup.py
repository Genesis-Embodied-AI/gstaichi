# Optional environment variables supported by setup.py:
#   {DEBUG, RELWITHDEBINFO, MINSIZEREL}
#     build the C++ gs_taichi_python extension with various build types.
#
#   TAICHI_CMAKE_ARGS
#     extra cmake args for C++ gs_taichi_python extension.

import glob
import multiprocessing
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from distutils.command.clean import clean
from distutils.dir_util import remove_tree

from setuptools import find_packages
from setuptools.command.develop import develop
from skbuild import setup
from skbuild.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel

root_dir = os.path.dirname(os.path.abspath(__file__))

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Topic :: Software Development :: Compilers",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Games/Entertainment :: Simulation",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]


def get_version():
    if os.getenv("RELEASE_VERSION"):
        version = os.environ["RELEASE_VERSION"]
    else:
        version_file = os.path.join(os.path.dirname(__file__), "version.txt")
        with open(version_file, "r") as f:
            version = f.read().strip()
    return version.lstrip("v")


project_name = os.getenv("PROJECT_NAME", "taichi")
version = get_version()
TI_VERSION_MAJOR, TI_VERSION_MINOR, TI_VERSION_PATCH = version.split(".")

data_files = glob.glob("python/_lib/runtime/*")
print(data_files)
packages = find_packages("python")
print(packages)

# Our python package root dir is python/
package_dir = "python"


def remove_tmp(taichi_dir):
    shutil.rmtree(os.path.join(taichi_dir, "assets"), ignore_errors=True)


class EggInfo(egg_info):
    def finalize_options(self, *args, **kwargs):
        if "" not in self.distribution.package_dir:
            # Issue#4975: skbuild loses the root package dir
            self.distribution.package_dir[""] = package_dir
        return super().finalize_options(*args, **kwargs)


def copy_assets():
    taichi_dir = os.path.join(package_dir, "gs_taichi")
    remove_tmp(taichi_dir)

    shutil.copytree("external/assets", os.path.join(taichi_dir, "assets"))


class Clean(clean):
    def run(self):
        super().run()
        self.build_temp = os.path.join(root_dir, "_skbuild")
        if os.path.exists(self.build_temp):
            remove_tree(self.build_temp, dry_run=self.dry_run)
        generated_folders = (
            "bin",
            "dist",
            "python/gs_taichi/assets",
            "python/gs_taichi/_lib/runtime",
            "python/gs_taichi/_lib/c_api",
            "gs_taichi.egg-info",
            "python/gs_taichi.egg-info",
            "build",
        )
        for d in generated_folders:
            if os.path.exists(d):
                remove_tree(d, dry_run=self.dry_run)
        generated_files = ["gs_taichi/common/commit_hash.h", "taichi/common/version.h"]
        generated_files += glob.glob("gs_taichi/runtime/llvm/runtime_*.bc")
        generated_files += glob.glob("python/gs_taichi/_lib/core/*.so")
        generated_files += glob.glob("python/gs_taichi/_lib/core/*.pyd")
        for f in generated_files:
            if os.path.exists(f):
                print(f"removing generated file {f}")
                if not self.dry_run:
                    os.remove(f)


def postprocess_stubs(stub_path: str) -> None:
    from ruamel.yaml import YAML

    stub_lines = stub_path.read_text().split("\n")
    yaml = YAML()
    with open("stub_replacements.yaml") as f:
        replacements = yaml.load(f)
    new_stub_lines = []
    for line in stub_lines:
        func_name = line.lstrip().partition("(")[0]
        if func_name in replacements:
            print("func_name", func_name)
            print("found func_name replacing with ", replacements[func_name])
            line = replacements[func_name]
        new_stub_lines.append(line)
    stub_path.write_text("\n".join(new_stub_lines))
    print("stub_path", stub_path)


def generate_pybind11_stubs(build_lib: str):
    build_lib_path = pathlib.Path(build_lib).resolve()
    taichi_path = build_lib_path.parent.parent / "cmake-install" / "python"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(taichi_path) + os.pathsep + env.get("PYTHONPATH", "")

    # command that works:
    # PYTHONPATH=_skbuild/linux-x86_64-3.10/cmake-install/python pybind11-stubgen \
    #     taichi._lib.core.gs_taichi_python --ignore-all-errors
    cmd_line = ["pybind11-stubgen", "gs_taichi._lib.core.gs_taichi_python", "--ignore-all-errors"]
    print(" ".join(cmd_line))
    subprocess.check_call(cmd_line, env=env)
    stub_filepath = pathlib.Path("stubs/gs_taichi/_lib/core/gs_taichi_python.pyi")
    postprocess_stubs(stub_filepath)

    target_filepath = build_lib_path / "gs_taichi" / "_lib" / "core" / "gs_taichi_python.pyi"
    py_typed_dst = build_lib_path / "gs_taichi" / "_lib" / "core" / "py.typed"
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    print("copying ", stub_filepath, "to", target_filepath)
    shutil.copy(stub_filepath, target_filepath)
    with open(py_typed_dst, "w"):
        pass  # creates an empty file


class DevelopWithStubs(develop):
    def run(self):
        super().run()
        build_lib = self.get_finalized_command("build_py").build_lib
        generate_pybind11_stubs(build_lib)


class BDistWheelWithStubs(bdist_wheel):
    def run(self):
        build_lib = self.get_finalized_command("build_py").build_lib
        generate_pybind11_stubs(build_lib)
        super().run()


def get_cmake_args():
    import shlex

    num_threads = os.getenv("BUILD_NUM_THREADS", multiprocessing.cpu_count())
    cmake_args = shlex.split(os.getenv("TAICHI_CMAKE_ARGS", "").strip())

    use_msbuild = False
    use_xcode = False

    if os.getenv("DEBUG", "0") in ("1", "ON"):
        cfg = "Debug"
    elif os.getenv("RELWITHDEBINFO", "0") in ("1", "ON"):
        cfg = "RelWithDebInfo"
    elif os.getenv("MINSIZEREL", "0") in ("1", "ON"):
        cfg = "MinSizeRel"
    else:
        cfg = None
    build_options = []
    if cfg:
        build_options.extend(["--build-type", cfg])
    if sys.platform == "win32":
        if os.getenv("TAICHI_USE_MSBUILD", "0") in ("1", "ON"):
            use_msbuild = True
        if use_msbuild:
            build_options.extend(["-G", "Visual Studio 17 2022"])
        else:
            build_options.extend(["-G", "Ninja", "--skip-generator-test"])
    if sys.platform == "darwin":
        if os.getenv("TAICHI_USE_XCODE", "0") in ("1", "ON"):
            use_xcode = True
        if use_xcode:
            build_options.extend(["-G", "Xcode", "--skip-generator-test"])
    sys.argv[2:2] = build_options

    cmake_args += [
        f"-DTI_VERSION_MAJOR={TI_VERSION_MAJOR}",
        f"-DTI_VERSION_MINOR={TI_VERSION_MINOR}",
        f"-DTI_VERSION_PATCH={TI_VERSION_PATCH}",
    ]

    if sys.platform == "darwin" and use_xcode:
        os.environ["SKBUILD_BUILD_OPTIONS"] = f"-jobs {num_threads}"
    elif sys.platform != "win32":
        os.environ["SKBUILD_BUILD_OPTIONS"] = f"-j{num_threads}"
    elif use_msbuild:
        # /M uses multi-threaded build (similar to -j)
        os.environ["SKBUILD_BUILD_OPTIONS"] = f"/M"
    if sys.platform == "darwin":
        if platform.machine() == "arm64":
            cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=arm64"]
        else:
            cmake_args += ["-DCMAKE_OSX_ARCHITECTURES=x86_64"]
    return cmake_args


# Control files to be included in package data
BLACKLISTED_FILES = [
    "libSPIRV-Tools-shared.so",
    "libSPIRV-Tools-shared.dll",
]

WHITELISTED_FILES = [
    "libMoltenVK.dylib",
]


def cmake_install_manifest_filter(manifest_files):
    def should_include(f):
        basename = os.path.basename(f)
        if basename in WHITELISTED_FILES:
            return True
        if basename in BLACKLISTED_FILES:
            return False
        return f.endswith((".so", "pyd", ".dll", ".bc", ".h", ".dylib", ".cmake", ".hpp", ".lib"))

    return [f for f in manifest_files if should_include(f)]


def sign_development_for_apple_m1():
    """
    Apple enforces codesigning for arm64 targets even for local development
    builds. See discussion here:
        https://github.com/supercollider/supercollider/issues/5603
    """
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            for path in glob.glob("python/gs_taichi/_lib/core/*.so"):
                print(f"signing {path}..")
                subprocess.check_call(["codesign", "--force", "--deep", "--sign", "-", path])
            for path in glob.glob("python/gs_taichi/_lib/c_api/lib/*.so"):
                print(f"signing {path}..")
                subprocess.check_call(["codesign", "--force", "--deep", "--sign", "-", path])
        except:
            print("cannot sign python shared library for macos arm64 build")


copy_assets()

force_plat_name = os.getenv("TAICHI_FORCE_PLAT_NAME", "").strip()
if force_plat_name:
    from skbuild.constants import set_skbuild_plat_name

    set_skbuild_plat_name(force_plat_name)

setup(
    name=project_name,
    packages=packages,
    package_dir={"": package_dir},
    version=version,
    description="GS Taichi Programming Language",
    author="GS Taichi developers",
    url="https://github.com/genesis-company/taichi",
    python_requires=">=3.10,<4.0",
    install_requires=[
        "numpy",
        "colorama",
        "dill",
        "rich",
        "setuptools>=68.0.0",  # Required for Python 3.12+ compatibility
        "cffi>=1.16.0",
    ],
    data_files=[
        (os.path.join("_lib", "runtime"), data_files),
    ],
    package_data={
        "gs_taichi._lib.core": ["gs_taichi_python.pyi", "py.typed"],
    },
    keywords=["graphics", "simulation"],
    license="Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ti=taichi._main:main",
        ],
    },
    classifiers=classifiers,
    cmake_args=get_cmake_args(),
    cmake_process_manifest_hook=cmake_install_manifest_filter,
    cmdclass={
        "egg_info": EggInfo,
        "clean": Clean,
        "bdist_wheel": BDistWheelWithStubs,
        "develop": DevelopWithStubs,
    },
    has_ext_modules=lambda: True,
)

sign_development_for_apple_m1()
