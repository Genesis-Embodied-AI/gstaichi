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


# ========== Tests for returns inside if statements ==========

# Kernel tests - void return
@pytest.mark.parametrize("in_val,expected", [
    (0, 2),
    (1, 5),
])
@test_utils.test()
def test_kernel_if_return_void_true_branch(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
        a[1] = 5

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 5),
    (1, 2),
])
@test_utils.test()
def test_kernel_if_return_void_false_branch(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 5
        else:
            a[1] = 2
            return
        a[1] = 10  # Should not execute

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 2),
    (1, 3),
])
@test_utils.test()
def test_kernel_if_return_void_both_branches(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
        else:
            a[1] = 3
            return
        a[1] = 99  # Should never execute

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


# Kernel tests - value return
@pytest.mark.parametrize("in_val,expected", [
    (0, 10),
    (1, 20),
])
@test_utils.test()
def test_kernel_if_return_value_true_branch(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            a[1] = 2
            return 10
        a[1] = 5
        return 20

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 20),
    (1, 10),
])
@test_utils.test()
def test_kernel_if_return_value_false_branch(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            a[1] = 5
            return 20
        else:
            a[1] = 2
            return 10
        return 99  # Should not execute

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 10),
    (1, 20),
])
@test_utils.test()
def test_kernel_if_return_value_both_branches(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            return 10
        else:
            return 20

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


# Function tests - void return
@pytest.mark.parametrize("in_val,expected", [
    (0, 2),
    (1, 5),
])
@test_utils.test()
def test_func_if_return_void_true_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
        a[1] = 5

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 5),
    (1, 2),
])
@test_utils.test()
def test_func_if_return_void_false_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 5
        else:
            a[1] = 2
            return
        a[1] = 10  # Should not execute

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 2),
    (1, 3),
])
@test_utils.test()
def test_func_if_return_void_both_branches(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
        else:
            a[1] = 3
            return
        a[1] = 99  # Should never execute

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


# Function tests - value return
@pytest.mark.parametrize("in_val,expected", [
    (0, 10),
    (1, 20),
])
@test_utils.test()
def test_func_if_return_value_true_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            a[1] = 2
            return 10
        a[1] = 5
        return 20

    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        return f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 20),
    (1, 10),
])
@test_utils.test()
def test_func_if_return_value_false_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            a[1] = 5
            return 20
        else:
            a[1] = 2
            return 10
        return 99  # Should not execute

    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        return f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 10),
    (1, 20),
])
@test_utils.test()
def test_func_if_return_value_both_branches(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            return 10
        else:
            return 20

    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        return f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


# Multiple if statements tests
@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_kernel_multiple_ifs_return_void(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 1
            return
        if a[0] == 1:
            a[1] = 2
            return
        a[1] = 3

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_kernel_multiple_ifs_return_value(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            return 1
        if a[0] == 1:
            return 2
        return 3

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_func_multiple_ifs_return_void(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 1
            return
        if a[0] == 1:
            a[1] = 2
            return
        a[1] = 3

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_func_multiple_ifs_return_value(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            return 1
        if a[0] == 1:
            return 2
        return 3

    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        return f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected


# Nested if tests
@pytest.mark.parametrize("in_val1,in_val2,expected", [
    (0, 0, 1),
    (0, 1, 2),
    (1, 0, 3),
    (1, 1, 4),
])
@test_utils.test()
def test_kernel_nested_ifs_return_void(in_val1: int, in_val2: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            if a[1] == 0:
                a[2] = 1
                return
            a[2] = 2
            return
        if a[1] == 0:
            a[2] = 3
            return
        a[2] = 4

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val1
    a[1] = in_val2
    k1(a)
    assert a[2] == expected


@pytest.mark.parametrize("in_val1,in_val2,expected", [
    (0, 0, 1),
    (0, 1, 2),
    (1, 0, 3),
    (1, 1, 4),
])
@test_utils.test()
def test_kernel_nested_ifs_return_value(in_val1: int, in_val2: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            if a[1] == 0:
                return 1
            return 2
        if a[1] == 0:
            return 3
        return 4

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val1
    a[1] = in_val2
    result = k1(a)
    assert result == expected


# Edge case: return with code after in same branch
@pytest.mark.parametrize("in_val,expected", [
    (0, 2),
    (1, 5),
])
@test_utils.test()
def test_kernel_if_return_void_with_code_after(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 2
            return
            a[1] = 99  # Should not execute
        a[1] = 5

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


# Edge case: if-elif-else with returns
@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_kernel_if_elif_else_return_void(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 1
            return
        elif a[0] == 1:
            a[1] = 2
            return
        else:
            a[1] = 3
            return

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize("in_val,expected", [
    (0, 1),
    (1, 2),
    (2, 3),
])
@test_utils.test()
def test_kernel_if_elif_else_return_value(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> ti.i32:
        if a[0] == 0:
            return 1
        elif a[0] == 1:
            return 2
        else:
            return 3

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    result = k1(a)
    assert result == expected
