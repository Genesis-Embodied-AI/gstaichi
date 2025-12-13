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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 5),
    ],
)
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 5),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 2),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, print_kernel_llvm_ir=True)
def test_kernel_if_return_void_false_branch(in_val: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        if a[0] == 0:
            a[1] = 5
        else:
            a[1] = 2
            return
        a[1] = 10

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 3),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 20),
    ],
)
@pytest.mark.xfail(reason="not handling return type yet")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 20),
        (1, 10),
    ],
)
@pytest.mark.xfail(reason="not handling return type yet")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 20),
    ],
)
@pytest.mark.xfail(reason="not handling return type yet")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 5),
    ],
)
@test_utils.test(offline_cache=False, advanced_optimization=True, print_kernel_llvm_ir=True)
def test_func_if_return_void_true_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> None:
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 2),
    ],
)
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_if_return_void_false_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> None:
        if a[0] == 0:
            a[1] = 5
        else:
            a[1] = 2
            return
        a[1] = 10

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        f1(a)

    a = ti.ndarray(ti.i32, (10,))
    a[0] = in_val
    k1(a)
    assert a[1] == expected


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 3),
    ],
)
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_if_return_void_both_branches(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> None:
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 20),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for returning value")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_if_return_value_true_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> ti.i32:
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 20),
        (1, 10),
    ],
)
@pytest.mark.xfail(reason="not handling returning values")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_if_return_value_false_branch(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> ti.i32:
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 10),
        (1, 20),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for returning value")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_if_return_value_both_branches(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> ti.i32:
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@pytest.mark.xfail(reason="not handling return type yet")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_func_multiple_ifs_return_void(in_val: int, expected: int):
    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1]) -> None:
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for returning value")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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
@pytest.mark.parametrize(
    "in_val1,in_val2,expected",
    [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 3),
        (1, 1, 4),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val1,in_val2,expected",
    [
        (0, 0, 1),
        (0, 1, 2),
        (1, 0, 3),
        (1, 1, 4),
    ],
)
@pytest.mark.xfail(reason="not handling return type yet")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 2),
        (1, 5),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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
@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


@pytest.mark.parametrize(
    "in_val,expected",
    [
        (0, 1),
        (1, 2),
        (2, 3),
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
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


# ========== Tests for returns inside for loops ==========


# For loop with if-return in true branch
@pytest.mark.parametrize(
    "stop_at,expected_sum",
    [
        (3, 3),  # 0 + 1 + 2, stops at i=3
        (5, 10),  # 0 + 1 + 2 + 3 + 4, stops at i=5
        (10, 45),  # Full range 0-9
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_for_if_return_void_early_exit(stop_at: int, expected_sum: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for i in range(10):
            if i == a[0]:
                return
            a[1] += i

    a = ti.ndarray(ti.i32, (10,))
    a[0] = stop_at
    a[1] = 0
    k1(a)
    assert a[1] == expected_sum


# For loop with if-return in false branch
@pytest.mark.parametrize(
    "skip_val,expected_sum",
    [
        (5, 40),  # Sum 0-9 except 5
        (0, 45),  # Sum 1-9 (skip 0)
        (9, 36),  # Sum 0-8 (skip 9)
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_for_if_else_return_void(skip_val: int, expected_sum: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for i in range(10):
            if i == a[0]:
                a[1] += 0  # Skip this value
            else:
                a[1] += i
                if i == 9:
                    return

    a = ti.ndarray(ti.i32, (10,))
    a[0] = skip_val
    a[1] = 0
    k1(a)
    assert a[1] == expected_sum


# Nested for loops with return
@pytest.mark.parametrize(
    "stop_i,stop_j,expected",
    [
        (2, 2, 22),  # thread 0: +5, thread 1: +5, thread 2: +2, htread 3: +5, thread 4: +5
        # total: 5 * 4 + 2 = 22
        (1, 1, 2),  # Stops at i=1, j=1
        (3, 3, 12),  # Various nested iterations
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_nested_for_if_return_void(stop_i: int, stop_j: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for i in range(5):
            for j in range(5):
                if i == a[0] and j == a[1]:
                    return
                print("i", i, "j", j)
                a[2] += i

    a = ti.ndarray(ti.i32, (10,))
    a[0] = stop_i
    a[1] = stop_j
    a[2] = 0
    k1(a)
    assert a[2] == expected


# Multiple if statements inside for loop with returns
@pytest.mark.parametrize(
    "threshold1,threshold2,expected_count",
    [
        (3, 7, 3),  # Counts 0, 1, 2 (stops at 3)
        (5, 7, 5),  # Counts 0-4 (stops at 5)
        (10, 7, 7),  # Counts 0-6 (stops at 7)
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_for_multiple_ifs_return_void(threshold1: int, threshold2: int, expected_count: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for i in range(10):
            if i == a[0]:
                return
            a[2] += 1
            if i == a[1]:
                return

    a = ti.ndarray(ti.i32, (10,))
    a[0] = threshold1
    a[1] = threshold2
    a[2] = 0
    k1(a)
    assert a[2] == expected_count


# For loop with if-return and code after loop
@pytest.mark.parametrize(
    "stop_at,expected_sum,expected_marker",
    [
        (3, 3, 0),  # Early return, marker not set
        (10, 45, 999),  # No early return, marker set
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_for_if_return_with_code_after_loop(stop_at: int, expected_sum: int, expected_marker: int):
    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for i in range(10):
            if i == a[0]:
                return
            a[1] += i
        a[2] = 999  # Only executes if loop completes

    a = ti.ndarray(ti.i32, (10,))
    a[0] = stop_at
    a[1] = 0
    a[2] = 0
    k1(a)
    assert a[1] == expected_sum
    assert a[2] == expected_marker


# Grouped field iteration with return
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_grouped_for_if_return_void():
    n_grids = 5
    g_field = ti.field(int, (n_grids, n_grids))
    count = ti.field(int, ())

    @ti.kernel
    def k1():
        for I in ti.grouped(g_field):
            if I[0] == 2 and I[1] == 2:
                return
            count[None] += 1

    count[None] = 0
    k1()
    # Should count all cells up to (2, 2)
    # (0,0) through (2,1) = 2*5 + 2 = 12 cells
    assert count[None] == 12


# Range for with dynamic bounds and return
@pytest.mark.parametrize(
    "n,threshold,expected",
    [
        (10, 5, 10),  # Sum 0+1+2+3+4
        (10, 10, 45),  # Full sum
        (10, 0, 0),  # Immediate return
    ],
)
@pytest.mark.xfail(reason="not implemented yet for kernels")
@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_kernel_dynamic_range_for_if_return_void(n: int, threshold: int, expected: int):
    @ti.kernel
    def k1(a: ti.types.NDArray, n: ti.i32) -> None:
        for i in range(n):
            if i == a[0]:
                return
            a[1] += i

    a = ti.ndarray(ti.i32, (10,))
    a[0] = threshold
    a[1] = 0
    k1(a, n)
    assert a[1] == expected


@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_if_func_for_loop_return_void() -> None:
    N = 4

    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1], b: ti.i32) -> None:
        print("b", b)
        if a[0] == 0:
            a[1 + b] = 2
            return
        a[1 + b] = 5

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for b in range(N):
            print("b", b)
            f1(a, b)

    a = ti.ndarray(ti.i32, (10,))
    for i in range(2):
        expected = {
            0: 2,
            1: 5,
        }[i]
        print("")
        print("=========")
        a[0] = i
        k1(a)
        print("a", a.to_numpy())
        for j in range(N):
            assert a[1 + j] == expected
            print("res i", i, f"a[{1 + j}]", a[1 + j])


@test_utils.test(offline_cache=False, advanced_optimization=False, print_kernel_llvm_ir=True)
def test_if_func_inner_loop_return_void() -> None:
    """Test return inside inner loop of ti.func - should continue kernel loop, not inner func loop"""
    N = 4

    @ti.func
    def f1(a: ti.types.NDArray[ti.i32, 1], b: ti.i32) -> None:
        for j in range(3):  # Inner loop in function
            if a[0] == 0:
                a[1 + b] = 2
                return  # Should continue kernel's b loop, not this j loop
        a[1 + b] = 5

    @ti.kernel
    def k1(a: ti.types.NDArray) -> None:
        for b in range(N):
            f1(a, b)

    a = ti.ndarray(ti.i32, (10,))
    for i in range(2):
        expected = {
            0: 2,
            1: 5,
        }[i]
        print("")
        print("=========")
        a[0] = i
        k1(a)
        print("a", a.to_numpy())
        for j in range(N):
            assert a[1 + j] == expected
            print("res i", i, f"a[{1 + j}]", a[1 + j])
