import pathlib
import sys

import pytest

import pytest

import gstaichi as ti
from gstaichi._test_tools import ti_init_same_arch

from tests import test_utils


@test_utils.test()
def test_src_ll_cache1(tmp_path: pathlib.Path) -> None:
    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.kernel
    def no_pure() -> None:
        pass

    no_pure()
    assert not no_pure._primal.src_ll_cache_observations.cache_key_generated

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    @ti.pure
    @ti.kernel
    def has_pure() -> None:
        pass

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert not has_pure._primal.src_ll_cache_observations.cache_validated
    assert not has_pure._primal.src_ll_cache_observations.cache_loaded
    assert has_pure._primal.src_ll_cache_observations.cache_stored

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)

    has_pure()
    assert has_pure._primal.src_ll_cache_observations.cache_key_generated
    assert has_pure._primal.src_ll_cache_observations.cache_validated
    assert has_pure._primal.src_ll_cache_observations.cache_loaded


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


@pytest.mark.parametrize(
    "flag_value",
    [
        None,
        False,
        True,
    ],
)
@test_utils.test()
def test_src_ll_cache_flag(tmp_path: pathlib.Path, flag_value: bool) -> None:
    """
    Test ti.init(src_ll_cache) flag
    """
    if flag_value:
        ti_init_same_arch(offline_cache_file_path=str(tmp_path), src_ll_cache=flag_value)
    else:
        ti_init_same_arch()

    @ti.pure
    @ti.kernel
    def k1() -> None:
        pass

    k1()
    cache_used = k1._primal.src_ll_cache_observations.cache_key_generated
    if flag_value:
        assert cache_used == flag_value
    else:
        assert cache_used  # default
