import os
import pathlib
import subprocess
import sys

import pydantic
import pytest

import gstaichi as ti
import gstaichi.lang
from gstaichi._test_tools import ti_init_same_arch

from tests import test_utils

TEST_RAN = "test ran"


@test_utils.test()
def test_src_ll_cache1(tmp_path: pathlib.Path) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.kernel
    def no_pure() -> None:
        pass

    no_pure()
    assert no_pure._primal is not None
    assert not no_pure._primal.src_ll_cache_observations.cache_key_generated

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.pure
    @ti.kernel
    def has_pure() -> None:
        pass

    has_pure()
    assert has_pure._primal is not None
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    assert has_pure._primal.src_ll_cache_observations.cache_stored
    assert has_pure._primal._last_compiled_kernel_data is not None

    last_compiled_kernel_data_str = None
    if gstaichi.lang.impl.current_cfg().arch in [ti.cpu, ti.cuda]:
        # we only support _last_compiled_kernel_data on cpu and cuda
        # and it only changes anything on cuda anyway, because it affects the PTX
        # cache
        last_compiled_kernel_data_str = has_pure._primal._last_compiled_kernel_data._debug_dump_to_string()
        assert last_compiled_kernel_data_str is not None and last_compiled_kernel_data_str != ""

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded
    if gstaichi.lang.impl.current_cfg().arch in [ti.cpu, ti.cuda]:
        assert has_pure._primal._last_compiled_kernel_data._debug_dump_to_string() == last_compiled_kernel_data_str


# Should be enough to run these on cpu I think, and anything involving
# stdout/stderr capture is fairly flaky on other arch
@test_utils.test(arch=ti.cpu)
@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows stderr not working with capfd")
def test_src_ll_cache_arg_warnings(tmp_path: pathlib.Path, capfd) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    class RandomClass:
        pass

    @ti.pure
    @ti.kernel
    def k1(foo: ti.Template) -> None:
        pass

    k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][PARAM_INVALID]" in err
    assert RandomClass.__name__ in err
    assert "[FASTCACHE][INVALID_FUNC]" in err
    assert k1.__name__ in err

    @ti.kernel
    def not_pure_k1(foo: ti.Template) -> None:
        pass

    not_pure_k1(foo=RandomClass())
    _out, err = capfd.readouterr()
    assert "[FASTCACHE][PARAM_INVALID]" not in err
    assert RandomClass.__name__ not in err
    assert "[FASTCACHE][INVALID_FUNC]" not in err
    assert k1.__name__ not in err


@test_utils.test()
def test_src_ll_cache_repeat_after_load(tmp_path: pathlib.Path) -> None:
    """
    Check that repeatedly calling kernel actually works, c.f. was doing
    no-op for a bit.
    """
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.pure
    @ti.kernel
    def has_pure(a: ti.types.NDArray[ti.i32, 1]) -> None:
        a[0] += 1

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    a[0] = 5
    for i in range(3):
        has_pure(a)
        assert a[0] == 6 + i

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    a[0] = 5
    for i in range(3):
        has_pure(a)
        assert a[0] == 6 + i


@pytest.mark.parametrize("src_ll_cache", [None, False, True])
@test_utils.test()
def test_src_ll_cache_flag(tmp_path: pathlib.Path, src_ll_cache: bool) -> None:
    """
    Test ti.init(src_ll_cache) flag
    """
    if src_ll_cache:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), src_ll_cache=src_ll_cache)
    else:
        ti_init_same_arch()

    @ti.pure
    @ti.kernel
    def k1() -> None:
        pass

    k1()
    cache_used = k1._primal.src_ll_cache_observations.cache_key_generated
    if src_ll_cache:
        assert cache_used == src_ll_cache
    else:
        assert cache_used  # default


class TemplateParamsKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    a: int
    src_ll_cache: bool


def src_ll_cache_template_params_child(args: list[str]) -> None:
    print("src_ll_cache_template_params_child")
    print("args", args)
    args_obj = TemplateParamsKernelArgs.model_validate_json(args[0])
    ti.init(
        arch=getattr(ti, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=args_obj.src_ll_cache,
    )
    print("after init")

    @ti.pure
    @ti.kernel
    def k1(a: ti.template(), output: ti.types.NDArray[ti.i32, 1]) -> None:
        output[0] = a

    print("after kernel declare")
    output = ti.ndarray(ti.i32, (10,))
    print("after create output. running kernel...")
    k1(args_obj.a, output)
    print("after running kernel")
    assert output[0] == args_obj.a
    print(TEST_RAN)


@pytest.mark.parametrize("src_ll_cache", [False, True])
@test_utils.test()
def test_src_ll_cache_template_params(tmp_path: pathlib.Path, src_ll_cache: bool) -> None:
    """
    template primitive kernel params should be in the cache key
    """
    arch = ti.lang.impl.current_cfg().arch.name

    def create_args(a: int) -> str:
        obj = TemplateParamsKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path),
            src_ll_cache=src_ll_cache,
            a=a,
        )
        json = TemplateParamsKernelArgs.model_dump_json(obj)
        return json

    env = os.environ
    env["PYTHONPATH"] = "."
    for a in [3, 4]:
        assert TEST_RAN in subprocess.check_output(
            [sys.executable, __file__, src_ll_cache_template_params_child.__name__, create_args(a)]
        ).decode("utf-8")


# The following lines are critical for the tests to work. If they are missing, the test will
# incorrectly pass, without doing anything.
if __name__ == "__main__":
    print("__main__")
    globals()[sys.argv[1]](sys.argv[2:])
