import pytest

import taichi as ti
from tests import test_utils


@pytest.mark.parametrize(
    "use_ndarray",
    [
        False,
        True,
    ],
)
@test_utils.test()
def test_class_method(use_ndarray: bool) -> None:
    if not use_ndarray:
        V = ti.field
        V_ANNOTATION = ti.template()
    else:
        V = ti.ndarray
        V_ANNOTATION = ti.types.ndarray()

    shape = (20,)

    @ti.data_oriented
    class InnerClass:
        def __init__(self):
            self.a = V(ti.i32, shape)
            self.b = V(ti.i32, shape)

    @ti.data_oriented
    class MyClass:
        def __init__(self):
            self.inner = InnerClass()
            self.c = V(ti.i32, shape)
            self.d = V(ti.i32, shape)
            self.e = V(ti.i32, shape)

        @ti.func
        def f1(self_unused, c: V_ANNOTATION, inner: ti.template()):
            c[0] = 4

        @ti.kernel
        def test(self, c: V_ANNOTATION, inner: ti.template()):
            self.f1(c, inner=inner)

        def run(self):
            self.test(self.c, self.inner)

    obj = MyClass()
    obj.run()
