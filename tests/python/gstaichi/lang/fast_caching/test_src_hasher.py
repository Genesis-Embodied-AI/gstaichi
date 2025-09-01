import pathlib
import shutil
from typing import Callable

import pytest

import gstaichi as ti
from gstaichi._test_tools import ti_init_same_arch
from gstaichi.lang import _wrap_inspect
from gstaichi.lang._fast_caching import function_hasher, src_hasher
from gstaichi.lang._wrap_inspect import get_source_info_and_src

from tests import test_utils


@test_utils.test()
def test_src_hasher_create_cache_key_vary_config() -> None:
    @ti.kernel
    def f1() -> None:
        pass

    # for some reason, print_ir_dbg_info is being set to True after a bit
    # so we are forcing it to false each initialization for now
    ti_init_same_arch(print_ir_dbg_info=False)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_base = src_hasher.create_cache_key(kernel_info, [])

    ti_init_same_arch(print_ir_dbg_info=False)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_same = src_hasher.create_cache_key(kernel_info, [])

    ti_init_same_arch(print_ir_dbg_info=False, random_seed=123)
    kernel_info, _src = get_source_info_and_src(f1.fn)
    cache_key_diff = src_hasher.create_cache_key(kernel_info, [])

    assert cache_key_base == cache_key_same
    assert cache_key_same != cache_key_diff


@test_utils.test()
def test_src_hasher_create_cache_key_vary_fn(monkeypatch, temporary_module) -> None:
    test_files_path = "tests/python/gstaichi/lang/fast_caching/test_files"
    monkeypatch.syspath_prepend(test_files_path)

    def get_cache_key(name: str) -> _wrap_inspect.FunctionSourceInfo:
        mod = temporary_module(name)
        info, _src = _wrap_inspect.get_source_info_and_src(mod.f1.fn)
        cache_key = src_hasher.create_cache_key(info, [])
        return cache_key

    key_base = get_cache_key("f1_base")
    key_same = get_cache_key("f1_same")
    key_diff = get_cache_key("f1_diff")

    assert key_base is not None
    assert key_base != ""
    assert key_base == key_same
    assert key_base != key_diff


@test_utils.test()
def test_src_hasher_validate_hashed_function_infos(monkeypatch, tmp_path: pathlib.Path, temporary_module) -> None:
    test_files_path = pathlib.Path("tests/python/gstaichi/lang/fast_caching/test_files")
    monkeypatch.syspath_prepend(tmp_path)

    def setup_folder(filename: str) -> None:
        shutil.copy2(test_files_path / filename, tmp_path / "child_diff.py")

    setup_folder("child_diff_base.py")
    mod = temporary_module("child_diff")

    def get_fileinfos(functions: list[Callable]) -> list[_wrap_inspect.FunctionSourceInfo]:
        fileinfos = []
        for f in functions:
            file_info, _src = _wrap_inspect.get_source_info_and_src(f)
            fileinfos.append(file_info)
        return fileinfos

    fileinfos = get_fileinfos([mod.f1.fn, mod.f2.fn])
    hashed_fileinfos = function_hasher.hash_functions(fileinfos)
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_same.py")
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_diff.py")
    assert not function_hasher.validate_hashed_function_infos(hashed_fileinfos)

    setup_folder("child_diff_same.py")
    assert function_hasher.validate_hashed_function_infos(hashed_fileinfos)


@test_utils.test()
def test_src_hasher_store_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, temporary_module) -> None:
    test_files_path = pathlib.Path("tests/python/gstaichi/lang/fast_caching/test_files")

    offline_cache_path = tmp_path / "cache"
    temp_import_path = tmp_path / "temp_import"
    temp_import_path.mkdir(exist_ok=True)

    ti_init_same_arch(offline_cache_file_path=str(offline_cache_path))

    monkeypatch.syspath_prepend(temp_import_path)

    def setup_folder(filename: str) -> None:
        shutil.copy2(test_files_path / filename, temp_import_path / "child_diff_test_src_hasher_store_validate.py")

    def get_fileinfos(functions: list[Callable]) -> list[_wrap_inspect.FunctionSourceInfo]:
        fileinfos = []
        for f in functions:
            file_info, _src = _wrap_inspect.get_source_info_and_src(f)
            fileinfos.append(file_info)
        return fileinfos

    setup_folder("child_diff_base.py")
    mod = temporary_module("child_diff_test_src_hasher_store_validate")
    kernel_info = get_fileinfos([mod.f1.fn])[0]
    fileinfos = get_fileinfos([mod.f1.fn, mod.f2.fn])
    cache_key = src_hasher.create_cache_key(kernel_info, [])

    assert not src_hasher.validate_cache_key(cache_key)

    src_hasher.store(cache_key, fileinfos)
    assert src_hasher.validate_cache_key(cache_key)

    setup_folder("child_diff_same.py")
    assert src_hasher.validate_cache_key(cache_key)

    setup_folder("child_diff_diff.py")
    assert not src_hasher.validate_cache_key(cache_key)

    setup_folder("child_diff_same.py")
    assert src_hasher.validate_cache_key(cache_key)

    assert not src_hasher.validate_cache_key("abcdefg")


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=ti.cpu)
@pytest.mark.parametrize(
    "print_non_pure",
    [
        None,
        False,
        True,
    ],
)
def test_src_hasher_print_non_pure(tmp_path: pathlib.Path, print_non_pure: bool | None, capfd) -> None:
    """
    Test ti.init parameter print_non_pure, which should print non pure functions when enabled
    """
    if print_non_pure:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True, print_non_pure=print_non_pure)
    else:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.pure
    @ti.kernel
    def k1_pure() -> None:
        pass

    k1_pure()
    out, _err = capfd.readouterr()
    output_contains_not_pure = "[NOT_PURE]" in out
    assert not output_contains_not_pure

    @ti.kernel
    def not_pure_k1() -> None:
        pass

    not_pure_k1()
    out, _err = capfd.readouterr()
    output_contains_not_pure = "[NOT_PURE]" in out
    if output_contains_not_pure:
        assert not_pure_k1.__name__ in out
    if print_non_pure is None:
        assert not output_contains_not_pure
    else:
        assert output_contains_not_pure == print_non_pure
