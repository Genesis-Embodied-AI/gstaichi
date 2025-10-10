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
RET_SUCCESS = 42


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
    args_obj = TemplateParamsKernelArgs.model_validate_json(args[0])
    ti.init(
        arch=getattr(ti, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=args_obj.src_ll_cache,
    )

    @ti.pure
    @ti.kernel
    def k1(a: ti.template(), output: ti.types.NDArray[ti.i32, 1]) -> None:
        output[0] = a

    output = ti.ndarray(ti.i32, (10,))
    k1(args_obj.a, output)
    assert output[0] == args_obj.a
    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


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
        proc = subprocess.run(
            [sys.executable, __file__, src_ll_cache_template_params_child.__name__, create_args(a)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != RET_SUCCESS:
            print(proc.stdout)  # needs to do this to see error messages
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


class HasReturnKernelArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    src_ll_cache: bool
    return_something: bool
    expect_used_src_ll_cache: bool
    expect_src_ll_cache_hit: bool


def src_ll_cache_has_return_child(args: list[str]) -> None:
    args_obj = HasReturnKernelArgs.model_validate_json(args[0])
    ti.init(
        arch=getattr(ti, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=args_obj.src_ll_cache,
    )

    @ti.pure
    @ti.kernel
    def k1(a: ti.i32, output: ti.types.NDArray[ti.i32, 1]) -> bool:
        output[0] = a
        if ti.static(args_obj.return_something):
            return True

    output = ti.ndarray(ti.i32, (10,))
    if args_obj.return_something:
        assert k1(3, output)
        # Sanity check that the kernel actually ran, and did something.
        assert output[0] == 3
        assert k1._primal.src_ll_cache_observations.cache_key_generated == args_obj.expect_used_src_ll_cache
        assert k1._primal.src_ll_cache_observations.cache_loaded == args_obj.expect_src_ll_cache_hit
        assert k1._primal.src_ll_cache_observations.cache_validated == args_obj.expect_src_ll_cache_hit
    else:
        # Even though we only check when not loading from the cache
        # we won't ever be able to load from the cache, since it will have failed
        # to cache the first time. By induction, it will always raise.
        with pytest.raises(
            ti.GsTaichiSyntaxError, match="Kernel has a return type but does not have a return statement"
        ):
            k1(3, output)
    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@pytest.mark.parametrize("return_something", [False, True])
@pytest.mark.parametrize("src_ll_cache", [False, True])
@test_utils.test()
def test_src_ll_cache_has_return(tmp_path: pathlib.Path, src_ll_cache: bool, return_something: bool) -> None:
    assert ti.lang is not None
    arch = ti.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    # need to test what happens when loading from fast cache, so run several runs
    # - first iteration stores to cache
    # - second and third will load from cache
    for it in range(3):
        args_obj = HasReturnKernelArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path),
            src_ll_cache=src_ll_cache,
            return_something=return_something,
            expect_used_src_ll_cache=src_ll_cache,
            expect_src_ll_cache_hit=src_ll_cache and it > 0,
        )
        args_json = HasReturnKernelArgs.model_dump_json(args_obj)
        cmd_line = [sys.executable, __file__, src_ll_cache_has_return_child.__name__, args_json]
        proc = subprocess.run(
            cmd_line,
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != RET_SUCCESS:
            print(" ".join(cmd_line))
            print(proc.stdout)  # needs to do this to see error messages
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


@test_utils.test()
def test_src_ll_cache_self_arg_checked(tmp_path: pathlib.Path) -> None:
    """
    Check that modifiying primtiive values in a data oriented object does result
    in the kernel correctly recompiling to reflect those new values, even with pure on.
    """
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.data_oriented
    class MyDataOrientedChild:
        def __init__(self) -> None:
            self.b = 10

    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.a = 3
            self.child = MyDataOrientedChild()

        @ti.pure
        @ti.kernel
        def k1(self) -> tuple[ti.i32, ti.i32]:
            return self.a, self.child.b

    my_do = MyDataOriented()

    # weirdly, if I don't use the name to get the arch, then on Mac github CI, the value of
    # arch can change during the below execcution 🤔
    # TODO: figure out why this is happening, and/or remove arch from python config object (replace
    # with arch_name and arch_idx for example)
    arch = getattr(ti, ti.lang.impl.current_cfg().arch.name)

    # need to initialize up front, in order that config hash doesn't change when we re-init later
    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 5
    my_do.child.b = 20
    assert tuple(my_do.k1()) == (5, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 5
    assert tuple(my_do.k1()) == (5, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated

    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 7
    assert tuple(my_do.k1()) == (7, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.a = 7
    assert tuple(my_do.k1()) == (7, 20)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated

    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.child.b = 30
    assert tuple(my_do.k1()) == (7, 30)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert not my_do.k1._primal.src_ll_cache_observations.cache_validated

    ti.reset()
    ti.init(arch=arch, offline_cache_file_path=str(tmp_path), offline_cache=True)
    my_do.child.b = 30
    assert tuple(my_do.k1()) == (7, 30)
    assert my_do.k1._primal.src_ll_cache_observations.cache_key_generated
    assert my_do.k1._primal.src_ll_cache_observations.cache_validated


# The following lines are critical for subprocess-using tests to work. If they are missing, the tests will
# incorrectly pass, without doing anything.
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
