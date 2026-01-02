import gstaichi as ti

from tests import test_utils


@test_utils.test(require=ti.extension.data64)
def test_clock():
    a = ti.field(dtype=ti.i64, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(block_dim=1)
        for i in range(32):
            start = ti.clock()
            x = ti.random() * 0.5 + 0.5
            for j in range((i + 1) * 2000):
                x = ti.sin(x * 1.0001 + j * 1e-6) + 1.2345
            if x != 0:
                a[i] = ti.clock() - start

    foo()

    if ti.lang.impl.get_runtime().prog.config().arch == ti.cuda:
        for i in range(1, 31):
            assert a[i - 1] < a[i] < a[i + 1]
            assert -1 < a[i] / a[0] - (i + 1) < 1
    else:
        for i in range(1, 31):
            assert a[i] == 0
