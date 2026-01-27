from enum import IntEnum

import gstaichi as ti
from gstaichi.lang._perf_dispatch import NUM_WARMUP

from tests import test_utils


@test_utils.test()
def test_perf_dispatch() -> None:
    class ImplEnum(IntEnum):
        serial = 0
        a_shape0_lt2 = 1
        a_shape0_ge2 = 2

    @ti.perf_dispatch(get_geometry_hash=lambda a,  b: hash(a.shape + c.shape))
    def my_func1(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]):
        ...

    @my_func1.register
    @ti.kernel
    def my_func1_impl_serial(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        B = a.shape[0]
        ti.loop_config(serialize=True)
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.serial] = 1

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] < 2)
    @ti.kernel
    def my_func1_impl_a_shape0_lt_2(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        print('my_func1_impl_a_shape0_lt_2')
        B = a.shape[0]
        ti.loop_config(serialize=False)
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.a_shape0_lt2] = 1

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] >= 2)
    @ti.kernel
    def my_func1_impl_a_shape0_ge_2(a: ti.types.NDArray[ti.i32, 1], c: ti.types.NDArray[ti.i32, 1]) -> None:
        print('my_func1_impl_a_shape0_ge_2')
        B = a.shape[0]
        ti.loop_config(serialize=False)
        for b in range(B):
            a[b] = a[b] * b
            c[ImplEnum.a_shape0_ge2] = 1

    N = 10000000
    a = ti.ndarray(ti.i32, (N,))
    c = ti.ndarray(ti.i32, (10,))

    for it in range(3):
        a.fill(5)
        my_func1(a, c)
        assert (a.to_numpy()[:5] == [0, 5, 20, 45, 80]).all()
        if it <= NUM_WARMUP:
            assert c[ImplEnum.serial] == 1
            assert c[ImplEnum.a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 1
        else:
            assert c[ImplEnum.a_shape0_ge2] == 1
            assert c[ImplEnum.a_shape0_lt2] == 0
            assert c[ImplEnum.a_shape0_ge2] == 0
