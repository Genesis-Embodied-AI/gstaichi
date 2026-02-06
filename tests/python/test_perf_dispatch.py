import time
from enum import IntEnum
from typing import cast

import pytest

import gstaichi as ti
from gstaichi.lang._perf_dispatch import NUM_WARMUP, PerformanceDispatcher
from gstaichi.lang.exception import GsTaichiSyntaxError

from tests import test_utils


@test_utils.test()
def test_perf_dispatch_basic() -> None:
    class ImplEnum(IntEnum):
        serial = 0
        a_shape0_lt2 = 1
        a_shape0_ge2 = 2

    @ti.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]): ...

    @my_func1.register
    @ti.kernel
    def my_func1_impl_slow(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        B = a.shape[0]
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.serial] = 1
            time.sleep(0.05)

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] < 2)
    @ti.kernel
    def my_func1_impl_a_shape0_lt_2(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        B = a.shape[0]
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.a_shape0_lt2] = 1
            time.sleep(0.03)

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] >= 2)
    @ti.kernel
    def my_func1_impl_a_shape0_ge_2(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        B = a.shape[0]
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.a_shape0_ge2] = 1
            time.sleep(0.01)

    N = 10
    a = ti.ndarray(ti.i32, (N,))
    c = ti.ndarray(ti.i32, (10,))

    for it in range((NUM_WARMUP + 5)):
        c.fill(0)
        for _inner_it in range(2):  # 2 compatible kernels
            a.fill(5)
            my_func1(a, c)
            assert (a.to_numpy()[:5] == [0, 5, 10, 15, 20]).all()
        if it <= NUM_WARMUP:
            assert c[ImplEnum.serial] == 1
            assert c[ImplEnum.a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
        else:
            assert c[ImplEnum.a_shape0_ge2] == 1
            assert c[ImplEnum.a_shape0_lt2] == 0
            assert c[ImplEnum.serial] == 0
    speed_checker = cast(PerformanceDispatcher, my_func1)
    geometry = list(speed_checker._trial_count_by_kernel_idx_by_geometry_hash.keys())[0]
    for _kernel_impl_idx, trials in speed_checker._trial_count_by_kernel_idx_by_geometry_hash[geometry].items():
        assert trials == NUM_WARMUP + 1
    assert len(speed_checker._trial_count_by_kernel_idx_by_geometry_hash[geometry]) == 2


@test_utils.test()
def test_perf_dispatch_swap_annotation_order() -> None:
    @ti.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]): ...

    with pytest.raises(GsTaichiSyntaxError):

        @ti.kernel
        @my_func1.register
        def my_func1_impl_serial(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None: ...


@test_utils.test()
def test_perf_dispatch_annotation_mismatch() -> None:
    @ti.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]): ...

    with pytest.raises(GsTaichiSyntaxError):

        @ti.kernel
        @my_func1.register
        def my_func1_impl_impl1(c: ti.types.NDArray[ti.i32, 1]) -> None: ...

    with pytest.raises(GsTaichiSyntaxError):

        @ti.kernel
        @my_func1.register
        def my_func1_impl_impl1(
            a: ti.types.NDArray[ti.i32, 1], d: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]
        ) -> None: ...
