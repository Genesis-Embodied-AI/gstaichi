import numpy as np
import pytest

import gstaichi as ti

from tests import test_utils


@pytest.mark.parametrize("static_value", [False, True])
@pytest.mark.parametrize("is_inner", [False, True])
@pytest.mark.parametrize("use_field", [False, True])
@test_utils.test()
def test_for_static_if_iter_runs(use_field: bool, is_inner: bool, static_value: bool) -> None:
    # Note that we currently dont have a way to turn static range on/off using some kind of variable/parameter.
    # So, for now, we'll have one side as static range, and one side as non-static range.
    # Since the code itself treats either side identically (same code path except for choosing one or the other side),
    # whilst the test isn't ideal, it should give identical coverage to something more rigorous.
    # We can think about approaches to parametrizing the static range in the future (nop function, macro,
    # parametrizablle ti.static, parametrizable ti.range, etc...).
    B = 2
    N_left = 3
    N_right = 5

    V = ti.field if use_field else ti.ndarray
    V_ANNOT = ti.Template if use_field else ti.types.NDArray[ti.i32, 2]

    if is_inner:

        @ti.kernel
        def k1(a: V_ANNOT) -> None:
            for b in range(B):
                for i in range(N_left) if ti.static(static_value) else ti.static(range(N_right)):
                    a[b, i] = 1

    else:

        @ti.kernel
        def k1(a: V_ANNOT) -> None:
            for i in range(N_left) if ti.static(static_value) else ti.static(range(N_right)):
                a[0, i] = 1

    a = V(ti.i32, (B, 6))
    k1(a)

    def create_expected():
        a_expected = np.zeros(dtype=np.int32, shape=(B, 6))
        for b in range(B) if is_inner else range(1):
            for i in range(N_left) if static_value else range(N_right):
                a_expected[b, i] = 1
        return a_expected

    assert np.all(create_expected() == a.to_numpy())


@pytest.mark.parametrize("is_static", [False, True])
@test_utils.test()
def test_for_static_if_iter_static_ranges(is_static: bool) -> None:
    # See comments on test_for_static_if_iter_runs for discussion of testing static vs non static ranges.

    # In this test, we verify that the static side is really static, and that the non-static side is
    # really non-static, by adding a conditional break to each, and seeing if that causes compilation to fail.

    # Note that break is only valid in inner loops, so we only test the inner loop case.
    B = 2
    N_left = 3
    N_right = 5

    @ti.kernel
    def k1(break_threshold: ti.i32) -> None:
        for b in range(B):
            for i in ti.static(range(N_left)) if ti.static(is_static) else range(N_right):
                if i >= break_threshold:
                    break

    if is_static:
        with pytest.raises(ti.GsTaichiCompilationError, match="You are trying to `break` a static `for` loop"):
            k1(0)
    else:
        # Dynamic break is ok, since not static for range.
        k1(0)
