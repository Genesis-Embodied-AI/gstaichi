import pathlib

import pytest

import gstaichi as ti
from gstaichi._test_tools import ti_init_same_arch

from tests import test_utils


@test_utils.test()
def test_fe_ll_observations(tmp_path: pathlib.Path) -> None:
    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        a[0] += 1

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    assert not k1._primal.fe_ll_cache_observations.cache_hit
    k1(a)
    assert not k1._primal.fe_ll_cache_observations.cache_hit

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit

    ti_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    a = ti.ndarray(ti.i32, (10,))
    k1._primal.fe_ll_cache_observations.cache_hit = False
    k1(a)
    assert k1._primal.fe_ll_cache_observations.cache_hit


@test_utils.test()
def test_ensure_compiled_reports_function():
    @ti.kernel
    def my_cool_test_function(a: ti.types.NDArray[ti.types.vector(2, ti.i32), 2]):
        pass

    x = ti.Matrix.ndarray(2, 3, ti.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=my_cool_test_function.__qualname__,
    ):
        my_cool_test_function(x)


@test_utils.test()
def test_pure_kernel_parameter() -> None:
    arch = ti.lang.impl.current_cfg().arch
    ti.init(arch=arch, offline_cache=False, src_ll_cache=True)

    @ti.pure
    @ti.kernel
    def k1() -> None: ...

    @ti.kernel(pure=True)
    def k2() -> None: ...

    @ti.kernel
    def k3() -> None: ...

    @ti.kernel(pure=False)
    def k4() -> None: ...

    k1()
    assert k1._primal.src_ll_cache_observations.cache_key_generated
    k2()
    assert k2._primal.src_ll_cache_observations.cache_key_generated
    k3()
    assert not k3._primal.src_ll_cache_observations.cache_key_generated
    k4()
    assert not k4._primal.src_ll_cache_observations.cache_key_generated
