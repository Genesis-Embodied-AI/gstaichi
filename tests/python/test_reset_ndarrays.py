import gstaichi as ti
import sys
from tests import test_utils


def check_objs(name: str) -> None:
    import gc
    all_objects = gc.get_objects()
    scalar_ndarrays = [obj for obj in all_objects if type(obj).__name__ == 'ScalarNdarray']
    print("check_objs", name)
    print(f"Found {len(scalar_ndarrays)} ScalarNdarray objects in memory")
    for i, obj in enumerate(scalar_ndarrays):
        print(f"  Object {i}: id={id(obj)}, shape={getattr(obj, 'shape', 'unknown')}")


@test_utils.test()
def test_ndarray_reset() -> None:
    import gc

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
    import gc
    arch = getattr(ti, ti.cfg.arch.name)
    for n in range(1000):
        ti.init(arch=arch)
        gc.collect() 
        a = ti.ndarray(ti.i32, shape=(55,))
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))
        check_objs("1")
        b = ti.ndarray(ti.i32, shape=(57,))
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))
        check_objs("2")
        c = ti.ndarray(ti.i32, shape=(211,))
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))
        print("\ncreating z param >>>>")
        check_objs("3")
        z_param = ti.ndarray(ti.i32, shape=(223,))
        print(" <<< z param created\n")
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))
        check_objs("4")
        bar_param = ti.ndarray(ti.i32, shape=(227,))
        print('a refcount', sys.getrefcount(a), sys.getrefcount(a.arr))

        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"
            print('ref count', v, sys.getrefcount(v), sys.getrefcount(v.arr))
        print('')

        check_objs("before kernel")

        print('define kernel')
        @ti.kernel
        def k1(z_param2: ti.types.NDArray[ti.i32, 1]) -> None:
            z_param2[33] += 2

        import gc
        check_objs("after kernel")

        print('')
        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"
            print('ref count', v, sys.getrefcount(v), sys.getrefcount(v.arr))
        print('')
        print("call gc...")
        gc.collect()
        for v in [a, b, c, z_param, bar_param]:
            assert len(v.arr.shape) > 0, f"{v}"
            print('ref count', v, sys.getrefcount(v), sys.getrefcount(v.arr))
        print("call gc...")
        gc.collect()
        print("call kernel")
        k1(z_param)
