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

    with pytest.raises(ti.GsTaichiNameError):
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

    with pytest.raises(ti.GsTaichiNameError):
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

    with pytest.raises(ti.GsTaichiNameError):
        k2()
