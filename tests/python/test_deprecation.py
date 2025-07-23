import math
import tempfile

import pytest

import gs_taichi as ti
from taichi._lib import core as _ti_core
from tests import test_utils


@test_utils.test()
def test_remove_element_shape_scalar():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The element_shape argument for scalar is deprecated in v1.6.0, and is removed in v1.7.0. "
        "Please remove them.",
    ):
        sym_x = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "x", dtype=ti.f32, element_shape=())


@test_utils.test()
def test_remove_element_shape_ndarray_arg():
    with pytest.raises(
        ti.TaichiRuntimeError,
        match="The element_shape argument for ndarray is deprecated in v1.6.0, and it is removed in v1.7.0. "
        "Please use vector or matrix data type instead.",
    ):
        ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1, element_shape=(1,))


@test_utils.test()
def test_remove_is_is_not():
    with pytest.raises(ti.TaichiSyntaxError, match='Operator "is" in Taichi scope is not supported'):

        @ti.kernel
        def func():
            ti.static(1 is 2)

        func()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_deprecate_experimental_real_func():
    with pytest.warns(
        DeprecationWarning,
        match="ti.experimental.real_func is deprecated because it is no longer experimental. "
        "Use ti.real_func instead.",
    ):

        @ti.experimental.real_func
        def foo(a: ti.i32) -> ti.i32:
            s = 0
            for i in range(100):
                if i == a + 1:
                    return s
                s = s + i
            return s

        @ti.kernel
        def bar(a: ti.i32) -> ti.i32:
            return foo(a)

        assert bar(10) == 11 * 5
        assert bar(200) == 99 * 50
