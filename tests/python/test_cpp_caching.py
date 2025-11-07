import gstaichi as ti

from tests import test_utils


@test_utils.test()
def test_cpp_caching() -> None:
    @ti.kernel
    def k1(p1: ti.types.NDArray, p2: ti.types.NDArray, p3: ti.types.NDArray) -> None:
        p3[0] = p1[0]
        p3[1] = p2[0]
    
    p1 = ti.ndarray(ti.i32, shape=(10,))
    p2 = ti.ndarray(ti.i32, shape=(10,))
    p3 = ti.ndarray(ti.i32, shape=(10,))

    for it in range(3):
        print('')
        print("===================")
        print("it", it)
        p1[0] = 5
        p2[0] = 7
        k1(p1, p2, p3)
        assert p3[0] == 5
        assert p3[1] == 7
        p3[0] = 0
        p3[1] = 0
