import gstaichi as ti

from tests import test_utils


@test_utils.test(require=ti.extension.data64)
def test_clock():
    a = ti.field(dtype=ti.i64, shape=32)

    @ti.kernel
    def foo():
        ti.loop_config(parallelize=1, block_dim=1)
        for i in range(32):
            start = ti.clock()
            x = ti.random() * 0.5 + 0.5
            for j in range((i + 1) * 400000):
                x = ti.sin(2.0 * x + j) + 1.0
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
