# This is a test file. It just has to exist, to check that pyright works with it.

from typing import Literal

import gstaichi as ti

from tests import test_utils

ti.init(arch=ti.cpu)


@ti.kernel
def k1(a: ti.types.ndarray(), b: ti.types.NDArray, c: ti.types.NDArray[ti.i32, Literal[1]]) -> None: ...


@test_utils.test()
def test_ndarray_type():
    a = ti.ndarray(ti.i32, (10,))
    k1(a, a, a)
