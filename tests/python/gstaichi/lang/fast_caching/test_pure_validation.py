import enum
import math

import numpy as np
import pytest

import gstaichi as ti

from tests import test_utils


@test_utils.test()
def test_pure_validation_prim():
    a = 2

    @ti.kernel
    def k1():
        print(a)

    k1()

    @ti.pure
    @ti.kernel
    def k1b(a: ti.i32):
        print(a)

    k1b(a)

    @ti.pure
    @ti.kernel
    def k1c(a: ti.Template):
        print(a)

    k1c(a)

    @ti.pure
    @ti.kernel
    def k2():
        print(a)

    with pytest.raises(ti.GsTaichiCompilationError):
        k2()


@test_utils.test()
def test_pure_validation_field():
    a = ti.field(ti.i32, (10,))

    @ti.kernel
    def k1_f():
        print(a[0])

    k1_f()

    @ti.pure
    @ti.kernel
    def k1c_f(a: ti.Template):
        print(a[0])

    k1c_f(a)

    @ti.pure
    @ti.kernel
    def k2_f():
        print(a[0])

    with pytest.raises(ti.GsTaichiCompilationError):
        k2_f()


@test_utils.test()
def test_pure_validation_field_child():
    a = ti.field(ti.i32, (10,))

    @ti.func
    def k1_f():
        print(a[0])

    @ti.kernel
    def k1():
        k1_f()

    k1()

    @ti.func
    def k1c_f(a: ti.Template):
        print(a[0])

    @ti.pure
    @ti.kernel
    def k1c(a: ti.Template):
        k1c_f(a)

    k1c(a)

    @ti.func
    def k2_f():
        print(a[0])

    @ti.pure
    @ti.kernel
    def k2():
        k2_f()

    with pytest.raises(ti.GsTaichiCompilationError):
        k2()


@test_utils.test()
def test_pure_validation_data_oriented_not_param():
    # When the data oriented arrives into the kernel without being passed in as a parameter,
    # to the kernel, that's a pure violation
    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.b = ti.field(ti.i32, (10,))

    @ti.pure
    @ti.kernel
    def k1() -> None:
        my_do.b[0] = 5

    my_do = MyDataOriented()
    with pytest.raises(ti.GsTaichiCompilationError):
        k1()


@test_utils.test()
def test_pure_validation_data_oriented_as_param():
    # When the data oriented arrives into the kernel as a parameter,
    # to the kernel, that's ok
    @ti.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.b = ti.field(ti.i32, (10,))

    @ti.pure
    @ti.kernel
    def k1(my_data_oriented: ti.template()) -> None:
        my_data_oriented.b[0] = 5

    my_do = MyDataOriented()
    k1(my_do)


@test_utils.test()
def test_pure_validation_enum():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @ti.kernel(pure=True)
    def k1() -> ti.i32:
        return MyEnum.bar

    v = k1()
    assert v == 2


@test_utils.test()
def test_pure_validation_builtin_values_inf():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @ti.kernel(pure=True)
    def k1() -> ti.i32:
        return ti.math.inf

    v = k1()
    assert v > 1e8


@test_utils.test()
def test_pure_validation_builtin_values_ti_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @ti.kernel(pure=True)
    def k1() -> ti.f32:
        return ti.math.pi

    v = k1()
    assert int(k1() * 100) == 314


@test_utils.test()
def test_pure_validation_builtin_values_np_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @ti.kernel(pure=True)
    def k1() -> ti.f32:
        return np.pi

    v = k1()
    assert int(k1() * 100) == 314


@test_utils.test()
def test_pure_validation_builtin_values_math_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @ti.kernel(pure=True)
    def k1() -> ti.f32:
        return math.pi

    assert int(k1() * 100) == 314
