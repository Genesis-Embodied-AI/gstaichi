import pytest

import taichi as ti
from tests import test_utils


@ti.kernel
def some_kernel(a: ti.template, b: ti.Template) -> None:
    a[None] = b[None] + 2


@test_utils.test()
def test_template_no_braces():
    a = ti.field(int, shape=())
    b = ti.field(int, shape=())
    b[None] = 5
    some_kernel(a, b)
    assert a[None] == 5 + 2
