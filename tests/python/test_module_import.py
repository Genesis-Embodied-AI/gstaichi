from tests import test_utils

import taichi as myowntaichi


@test_utils.test()
def test_module_import():
    @myowntaichi.kernel
    def func():
        for _ in myowntaichi.static(range(8)):
            pass

    func()
