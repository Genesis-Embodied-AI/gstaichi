import gstaichi as ti
import sys
from tests import test_utils
import gc


def check_objs(name: str) -> None:
    gc.get_objects()


@test_utils.test()
def test_ndarray_reset() -> None:
    @ti.kernel
    def k1(z_param2: ti.types.NDArray[ti.i32, 1]) -> None:
        z_param2[33] += 2

    arch = getattr(ti, ti.cfg.arch.name)

    for n in range(1000):
        # ti.reset()
        # ti_init_same_arch()
        ti.init(arch=arch)
        n = ti.ndarray(ti.i32, (233,))
        n[0] = 3
        # print(1, n.arr.shape)
        assert len(n.arr.shape) > 0
        # k1(n)
        # print(1, n.arr.shape)
        assert len(n.arr.shape) > 0
        gc.collect()
        # print(2, n.arr.shape)
        assert len(n.arr.shape) > 0
        # k1(n)
        n[0] = 3
        # print(1, n.arr.shape)
        assert len(n.arr.shape) > 0


@test_utils.test()
def test_ndarray_simple_kernel_call() -> None:
    arch = getattr(ti, ti.cfg.arch.name)
    for n in range(1000):
        ti.init(arch=arch)
        gc.collect() 
        a = ti.ndarray(ti.i32, shape=(55,))
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))
        check_objs("1")
        b = ti.ndarray(ti.i32, shape=(57,))
        check_objs("2")
        c = ti.ndarray(ti.i32, shape=(211,))
        check_objs("3")
        z_param = ti.ndarray(ti.i32, shape=(223,))
        check_objs("4")
        bar_param = ti.ndarray(ti.i32, shape=(227,))

        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"

        check_objs("before kernel")

        @ti.kernel
        def k1(z_param2: ti.types.NDArray[ti.i32, 1]) -> None:
            z_param2[33] += 2

        check_objs("after kernel")

        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"
        gc.collect()
        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"
            print('ref count', v, sys.getrefcount(v), sys.getrefcount(v.arr))
        gc.collect()
        k1(z_param)
