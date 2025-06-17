import argparse
import copy
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from taichi._main import TaichiMain

import taichi as ti
from tests import test_utils


@contextmanager
def patch_sys_argv_helper(custom_argv: list):
    """Temporarily patch sys.argv for testing."""
    try:
        cached_argv = copy.deepcopy(sys.argv)
        sys.argv = custom_argv
        yield sys.argv
    finally:
        sys.argv = cached_argv


def test_cli_exit_one_with_no_command_provided():
    with patch_sys_argv_helper(["ti"]):
        cli = TaichiMain(test_mode=True)
        assert cli() == 1


def test_cli_exit_one_with_bogus_command_provided():
    with patch_sys_argv_helper(["ti", "bogus-command-not-registered-yet"]):
        cli = TaichiMain(test_mode=True)
        assert cli() == 1


def test_cli_can_dispatch_commands_to_methods_correctly():
    with patch_sys_argv_helper(["ti", "example", "bogus_example_name_for_test"]):
        with patch.object(TaichiMain, "example", return_value=None) as mock_method:
            cli = TaichiMain(test_mode=False)
            cli()
            mock_method.assert_called_once_with(["bogus_example_name_for_test"])


def test_cli_example():
    with patch_sys_argv_helper(["ti", "example", "minimal"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal"

    with patch_sys_argv_helper(["ti", "example", "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal"

    with patch_sys_argv_helper(["ti", "example", "-s", "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.save == True

    with patch_sys_argv_helper(["ti", "example", "-p", "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.print == True

    with patch_sys_argv_helper(["ti", "example", "-P", "minimal.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.name == "minimal" and args.pretty_print == True


def test_cli_regression():
    with patch_sys_argv_helper(["ti", "regression", "a.py", "b.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.files == ["a.py", "b.py"]


def test_cli_debug():
    with patch_sys_argv_helper(["ti", "debug", "a.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.filename == "a.py"


def test_cli_run():
    with patch_sys_argv_helper(["ti", "run", "a.py"]) as custom_argv:
        cli = TaichiMain(test_mode=True)
        args = cli()
        assert args.filename == "a.py"


def test_cli_cache():
    archs = {ti.cpu, ti.cuda, ti.opengl, ti.vulkan, ti.metal, ti.gles, ti.amdgpu}
    expected_archs = test_utils.expected_archs()
    archs = {v for v in archs if v in test_utils.expected_archs()}
    exts = ("tic", "tcb", "lock")
    tmp_path = tempfile.mkdtemp()

    @ti.kernel
    def simple_kernel(a: ti.i32) -> ti.i32:
        return -a

    def launch_kernel(arch):
        ti.init(arch=arch, offline_cache=True, offline_cache_file_path=tmp_path)
        simple_kernel(128)
        ti.reset()

    for arch in archs:
        launch_kernel(arch)

    found = False
    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            if file.endswith(exts):
                found = True
                break
        if found:
            break
    assert found

    with patch_sys_argv_helper(["ti", "cache", "clean", "-p", tmp_path]) as custom_argv:
        cli = TaichiMain()
        cli()

    for root, dirs, files in os.walk(tmp_path):
        for file in files:
            assert not file.endswith(exts)

    shutil.rmtree(tmp_path)
