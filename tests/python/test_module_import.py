import gstaichi as myowngstaichi
from tests import test_utils


@test_utils.test()
def test_module_import():
    @myowngstaichi.kernel
    def func():
        for _ in myowngstaichi.static(range(8)):
            pass

    func()
