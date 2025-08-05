import gstaichi as ti
from tests import test_utils


@ti.func
def some_sub_func(a: ti.template, b: ti.Template) -> None:
    a[None] = b[None] + 2


@ti.kernel
def some_kernel(a: ti.template, b: ti.Template) -> None:
    a[None] = b[None] + 2
    some_sub_func(a, b)


@test_utils.test()
def test_template_no_braces():
    """
    Check that we can use ti.Template as an annotation for kernels and funcs.
    """
    a = ti.field(int, shape=())
    b = ti.field(int, shape=())
    b[None] = 5
    some_kernel(a, b)
    assert a[None] == 5 + 2
