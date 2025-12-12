import pytest

import gstaichi as ti

from tests import test_utils


@test_utils.test()
def test_ifexpr_vector():
    n_grids = 10

    g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = ti.field(float, (n_grids, n_grids, n_grids))

    @ti.kernel
    def func():
        for I in ti.grouped(g_m):
            cond = (I < 3) & (g_v[I] < 0) | (I > n_grids - 3) & (g_v[I] > 0)
            g_v[I] = 0 if cond else g_v[I]

    with pytest.raises(ti.GsTaichiSyntaxError, match='Please use "ti.select" instead.'):
        func()


@test_utils.test()
def test_ifexpr_scalar():
    n_grids = 10

    g_v = ti.Vector.field(3, float, (n_grids, n_grids, n_grids))
    g_m = ti.field(float, (n_grids, n_grids, n_grids))

    @ti.kernel
    def func():
        for I in ti.grouped(g_m):
            cond = (I[0] < 3) and (g_v[I][0] < 0) or (I[0] > n_grids - 3) and (g_v[I][0] > 0)
            g_v[I] = 0 if cond else g_v[I]

    func()


@pytest.mark.parametrize("in_val,expected",[
    (0, 2),
    (1, 5),
])
@test_utils.test()
def test_if_return_void(in_val: int, expected: int) -> None:
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
        a[1] = 5

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected
