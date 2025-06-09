import pytest

from tests import test_utils

import taichi as ti


@test_utils.test()
def test_nested_kernel_error():
    @ti.kernel
    def B():
        pass

    @ti.kernel
    def A():
        B()

    with pytest.raises(ti.TaichiCompilationError):
        A()
